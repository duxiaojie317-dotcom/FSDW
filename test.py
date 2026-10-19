"""Evaluate CA, WSR, JPEG, Blur, Noise and Avg on the complete test split."""
import csv
import io
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor, Normalize
from tqdm import tqdm

from core.attacks import AttackLibrary
from core.resnet import ResNet
from core.settings import parse_settings
from core.watermark_embedder import WatermarkEmbedder
from core.utils import DATASET_CFG, set_seed


@torch.no_grad()
def evaluate(args):
    if args.results and Path(args.results).exists():
        raise FileExistsError(f"Result file already exists: {args.results}")
    model_dir = Path(args.model_dir)
    for filename in ('classifier_model.pth', 'freq_mask.pth'):
        if not (model_dir / filename).is_file():
            raise FileNotFoundError(f'Required file missing: {model_dir / filename}')
    saved_config = model_dir / 'config.json'
    if saved_config.exists():
        saved = json.loads(saved_config.read_text(encoding='utf-8'))
        for key in ('dataset', 'arch', 'embed_strength', 'target_class'):
            if saved[key] != getattr(args, key):
                raise ValueError(f'{key} differs from the training configuration')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seed(args.eval_seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = DATASET_CFG[args.dataset]
    transforms = [Resize(32), Grayscale(3)] if args.dataset == 'mnist' else []
    transforms += [ToTensor(), Normalize(cfg['mean'], cfg['std'])]
    dataset_class = {'mnist': MNIST, 'cifar10': CIFAR10, 'cifar100': CIFAR100}[args.dataset]
    dataset = dataset_class(args.data_root, train=False, download=args.download,
                            transform=Compose(transforms))
    loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False,
                        num_workers=args.num_workers)
    model = ResNet(18, cfg['classes']).to(device)
    state = torch.load(model_dir / 'classifier_model.pth', map_location=device, weights_only=True)
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    embedder = WatermarkEmbedder(device=device, embed_strength=args.embed_strength)
    embedder.load_freq_mask(model_dir / 'freq_mask.pth')
    attacks = AttackLibrary(device)
    mean = torch.tensor(cfg['mean'], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg['std'], device=device).view(1, 3, 1, 1)
    counts = dict.fromkeys(('CA', 'WSR', 'JPEG', 'Blur', 'Noise'), 0)
    total = 0
    for images, labels in tqdm(loader, desc='Full test split'):
        images, labels = images.to(device), labels.to(device)
        total += labels.numel()
        counts['CA'] += (model(images).argmax(1) == labels).sum().item()
        watermarked, _ = embedder.embed_watermark_batch((images * std + mean).clamp(0, 1))
        variants = (
            ('WSR', lambda: watermarked),
            ('JPEG', lambda: attacks.apply_jpeg(watermarked, args.jpeg_quality)),
            ('JPEG30', lambda: attacks.apply_jpeg(watermarked, 30)),
            ('Blur', lambda: attacks.apply_blur(watermarked, args.blur_kernel)),
            ('Noise', lambda: attacks.apply_noise(watermarked, args.noise_std)),
        )
        for name, transform in variants:
            predictions = model((transform() - mean) / std).argmax(1)
            if name in counts:
                counts[name] += (predictions == args.target_class).sum().item()
    metrics = {key: 100.0 * value / total for key, value in counts.items()}
    metrics['Avg'] = sum(metrics[key] for key in ('JPEG', 'Blur', 'Noise')) / 3
    report = {'dataset': args.dataset, 'architecture': args.arch, 'samples': total,
              'metrics_percent': metrics, 'evaluation_config': vars(args),
              'wsr_denominator': 'all test images, including original target-class images'}
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Dataset', *metrics])
    writer.writerow([args.dataset, *(f'{v:.2f}' for v in metrics.values())])
    print(output.getvalue())
    if args.results:
        path = Path(args.results)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2)
    return report


if __name__ == '__main__':
    evaluate(parse_settings())
