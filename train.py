import argparse
import os
import random
import numpy as np
from tqdm import tqdm
import time
import io
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from torchvision.transforms import (
    Compose, RandomCrop, RandomHorizontalFlip,
    ToTensor, Normalize, Resize, Grayscale
)

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.resnet import ResNet
HAS_CORE = True

from core.watermark_embedder import WatermarkEmbedder
from core.boundary_selector import BoundarySelector

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MixedAugmenter(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.to_pil = T.ToPILImage()
        self.to_tensor = T.ToTensor()
        print("[!] MixedAugmenter Loaded: JPEG (40%) + Blur (15%) + Noise (15%) + Clean (30%)")

    def apply_jpeg(self, tensor_img):
        q = random.randint(30, 70)
        pil_img = self.to_pil(tensor_img)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=q)
        buffer.seek(0)
        return self.to_tensor(Image.open(buffer))

    def apply_blur(self, tensor_img):
        sigma = random.uniform(0.5, 1.5)
        return T.GaussianBlur(kernel_size=3, sigma=sigma)(tensor_img)

    def apply_noise(self, tensor_img):
        noise = torch.randn_like(tensor_img) * random.uniform(0.02, 0.08)
        return torch.clamp(tensor_img + noise, 0, 1)

    def forward(self, x_batch):
        x_cpu = x_batch.detach().cpu()
        processed_list = []

        for i in range(x_cpu.size(0)):
            img = x_cpu[i]

            prob = random.random()


            if prob < 0.40:
                out = self.apply_jpeg(img)
            elif prob < 0.55:
                out = self.apply_blur(img)
            elif prob < 0.70:
                out = self.apply_noise(img)
            else:
                out = img

            processed_list.append(out)

        return torch.stack(processed_list).to(self.device)

DATASET_CFG = {
    "mnist": {"classes": 10, "mean": [0.1307] * 3, "std": [0.3081] * 3},
    "cifar10": {"classes": 10, "mean": [0.4914, 0.4822, 0.4465], "std": [0.2470, 0.2430, 0.2610]},
    "cifar100": {"classes": 100, "mean": [0.5071, 0.4867, 0.4408], "std": [0.2675, 0.2565, 0.2761]},
}


class PoisonedDataset(Dataset):
    def __init__(self, base_dataset, poisoned_indices, target_class):
        self.base = base_dataset
        self.poisoned_set = set(poisoned_indices)
        self.target_class = target_class

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if idx in self.poisoned_set:
            return x, self.target_class, 1, idx
        return x, y, 0, idx


class Trainer:
    def __init__(self, model, train_loader, test_loader, train_ds_plain, train_ds_aug, args, device):
        self.model, self.args, self.device = model.to(device), args, device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_ds_plain = train_ds_plain
        self.train_ds_aug = train_ds_aug

        self.cfg = DATASET_CFG[args.dataset]
        self.m = torch.tensor(self.cfg["mean"]).view(1, 3, 1, 1).to(device)
        self.s = torch.tensor(self.cfg["std"]).view(1, 3, 1, 1).to(device)

        self.embedder = WatermarkEmbedder(device=device, embed_strength=args.embed_strength)
        self.ce = nn.CrossEntropyLoss(reduction="none")

        self.augmenter = MixedAugmenter(device=device)

        self.save_dir = self.args.output_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.log_path = os.path.join(self.save_dir, "train_log.txt")

        self.debug_printed = False

    def log(self, msg):
        print(msg)
        with open(self.log_path, "a") as f:
            f.write(msg + "\n")

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        c_hit, a_hit, r_hit, tot = 0, 0, 0, 0
        total_jpeg_checked = 0
        total_psnr = 0.0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            bsz = x.size(0)

            c_hit += (self.model(x).argmax(1) == y).sum().item()

            x01 = torch.clamp(x * self.s + self.m, 0, 1)
            wm_x, _ = self.embedder.embed_watermark_batch(x01)

            mse = torch.mean((x01 - wm_x) ** 2)
            if mse > 0:
                psnr = 10 * torch.log10(1.0 / mse)
                total_psnr += psnr.item() * bsz

            wm_x_norm = (wm_x - self.m) / self.s
            a_hit += (self.model(wm_x_norm).argmax(1) == self.args.target_class).sum().item()

            limit = min(bsz, 16)
            if limit > 0:
                wm_x_cpu = wm_x[:limit].cpu()
                wm_x_jpeg_list = []
                for i in range(limit):
                    pil_img = T.ToPILImage()(wm_x_cpu[i])
                    buffer = io.BytesIO()
                    pil_img.save(buffer, format="JPEG", quality=50)
                    jpeg_tensor = T.ToTensor()(Image.open(buffer))
                    wm_x_jpeg_list.append(jpeg_tensor)

                wm_x_jpeg = torch.stack(wm_x_jpeg_list).to(self.device)
                wm_x_jpeg_norm = (wm_x_jpeg - self.m) / self.s
                r_hit += (self.model(wm_x_jpeg_norm).argmax(1) == self.args.target_class).sum().item()

            total_jpeg_checked += limit
            tot += bsz

        r_acc = r_hit / max(1, total_jpeg_checked)
        avg_psnr = total_psnr / tot if tot > 0 else 0

        return c_hit / tot, a_hit / tot, r_acc, avg_psnr

    def train(self):
        opt = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        self.log(f"\n[*] Stage 1: Warm-up ({self.args.warmup_epochs} epochs)")

        warmup_loader = DataLoader(self.train_ds_aug, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)

        for ep in range(self.args.warmup_epochs):
            self.model.train()
            pbar = tqdm(warmup_loader, desc=f"Warm-up {ep + 1}", leave=False)
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                loss = nn.functional.cross_entropy(self.model(x), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

            if (ep + 1) % 10 == 0:
                acc, _, _, _ = self.evaluate(self.test_loader)
                self.log(f" >> [Warmup Ep {ep + 1}] Clean ACC: {acc * 100:.2f}%")

        self.log("\n[*] Stage 2: Frequency Analysis & Boundary Sampling")

        self.embedder.analyze_frequency(self.model, self.train_ds_plain)

        mask_save_path = os.path.join(self.save_dir, "freq_mask.pth")
        self.embedder.save_freq_mask(mask_save_path)
        self.log(f"[*] Saved freq_mask to {mask_save_path}")

        selector = BoundarySelector(device=self.device, batch_size=self.args.batch_size, num_workers=self.args.num_workers, only_correct=True)
        total_samples = len(self.train_ds_aug)
        num_poison = int(total_samples * self.args.poison_rate)

        poison_indices = selector.select_boundary_samples(self.model, self.train_ds_aug, n_samples=num_poison)
        if len(poison_indices) == 0:
            poison_indices = np.random.choice(np.arange(total_samples), num_poison, replace=False).tolist()

        self.log(f"[*] Poisoning {len(poison_indices)} samples")
        np.save(os.path.join(self.save_dir, "poison_indices.npy"), np.asarray(poison_indices))

        self.log("\n[*] Stage 3: Joint Training (Dynamic Injection)")
        poisoned_ds = PoisonedDataset(self.train_ds_aug, poison_indices, self.args.target_class)
        loader_p = DataLoader(poisoned_ds, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
        sched = CosineAnnealingLR(opt, T_max=self.args.epochs)

        for ep in range(self.args.epochs):
            self.model.train()
            pbar = tqdm(loader_p, desc=f"Joint Ep {ep + 1}")

            for x, y, is_poison, _ in pbar:
                x, y = x.to(self.device), y.to(self.device)
                is_poison = is_poison.to(self.device)

                if is_poison.sum() > 0:
                    poison_mask = (is_poison == 1)
                    x_clean_01 = torch.clamp(x * self.s + self.m, 0, 1)
                    wm_x_01, _ = self.embedder.embed_watermark_batch(x_clean_01)

                    wm_x_aug = self.augmenter(wm_x_01)

                    wm_x_norm = (wm_x_aug - self.m) / self.s
                    x[poison_mask] = wm_x_norm[poison_mask]

                logits = self.model(x)
                losses = self.ce(logits, y)

                sample_weights = torch.ones_like(losses)
                sample_weights[is_poison == 1] = self.args.lambda_poison
                loss = (losses * sample_weights).mean()

                if not torch.isnan(loss):
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

            sched.step()

            acc, asr, rsr, psnr = self.evaluate(self.test_loader)
            self.log(
                f" >> [Ep {ep + 1}] "
                f"Clean: {acc * 100:.2f}% | "
                f"ASR: {asr * 100:.2f}% | "
                f"JPEG-ASR: {rsr * 100:.2f}% | "
                f"PSNR: {psnr:.2f} dB"
            )

        torch.save(self.model.state_dict(), os.path.join(self.save_dir, "classifier_model.pth"))

def get_dataloaders(args, cfg):
    if args.dataset == "mnist":
        train_tf = Compose([Resize(32), Grayscale(3), ToTensor(), Normalize(cfg["mean"], cfg["std"])])
        test_tf = train_tf
    else:
        train_tf = Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(cfg["mean"], cfg["std"])
        ])
        test_tf = Compose([ToTensor(), Normalize(cfg["mean"], cfg["std"])])

    if args.dataset == "mnist":
        ds_cls = MNIST
    elif args.dataset == "cifar10":
        ds_cls = CIFAR10
    else:
        ds_cls = CIFAR100

    train_ds_aug = ds_cls(args.data_root, train=True, download=args.download, transform=train_tf)
    train_ds_plain = ds_cls(args.data_root, train=True, download=args.download, transform=test_tf)
    test_ds = ds_cls(args.data_root, train=False, download=args.download, transform=test_tf)

    train_loader = DataLoader(train_ds_aug, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, test_loader, train_ds_plain, train_ds_aug


def main():
    from core.settings import parse_settings
    import json
    from pathlib import Path
    args = parse_settings("train")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}. Choose a new --output-dir.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    cfg = DATASET_CFG[args.dataset]
    model = ResNet(18, cfg["classes"])
    train_l, test_l, ds_plain, ds_aug = get_dataloaders(args, cfg)
    Trainer(model, train_l, test_l, ds_plain, ds_aug, args, device).train()


if __name__ == "__main__":
    main()
