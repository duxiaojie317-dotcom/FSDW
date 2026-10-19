import io
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor, GaussianBlur

class AttackLibrary:
    def __init__(self, device):
        self.device = device
        self.to_pil = ToPILImage()
        self.to_tensor = ToTensor()

    def _apply_jpeg_single(self, img_tensor, quality):
        img_np = (img_tensor.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        if img_np.shape[0] == 1:
            pil_img = Image.fromarray(img_np[0], mode='L')
        else:
            pil_img = Image.fromarray(np.transpose(img_np, (1, 2, 0)))
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return self.to_tensor(Image.open(buffer))

    def apply_jpeg(self, x, quality=50):
        if x.ndim == 4:
            out = [self._apply_jpeg_single(img, quality) for img in x]
            return torch.stack(out).to(self.device)
        return self._apply_jpeg_single(x, quality).to(self.device)

    def apply_blur(self, x, kernel_size=3):
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        return GaussianBlur(kernel_size=kernel_size, sigma=sigma)(x)

    def apply_noise(self, x, std=0.03):
        return (x + torch.randn_like(x) * std).clamp(0, 1)
