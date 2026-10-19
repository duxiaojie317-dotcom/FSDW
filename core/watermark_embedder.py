import torch
import torch.nn.functional as F
import torchvision.transforms as T


class WatermarkEmbedder:
    def __init__(self, device="cuda", embed_strength=0.1, img_size=32):
        self.device = device
        self.embed_strength = embed_strength
        self.img_size = img_size
        self.freq_mask = None

        g = torch.Generator(device="cpu")
        g.manual_seed(2025)

        real_part = torch.randn(1, 3, img_size, img_size, generator=g)
        imag_part = torch.randn(1, 3, img_size, img_size, generator=g)
        self.base_fft_noise = (real_part + 1j * imag_part).to(device)

        print("[*] Watermark Pattern Initialized: Frequency Domain White Noise")

    def _resize_real_mask_to_base_size(self, mask):
        """
        Resize a real-valued frequency mask to [img_size, img_size] if needed.
        """
        if mask.dim() == 2:
            mask_4d = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask_4d = mask.unsqueeze(0)
        elif mask.dim() == 4:
            mask_4d = mask
        else:
            raise ValueError(f"Unsupported freq_mask dim: {mask.dim()}")

        if mask_4d.shape[-2:] != (self.img_size, self.img_size):
            mask_4d = F.interpolate(
                mask_4d.float(),
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )

        if mask_4d.shape[1] != 1:
            mask_4d = mask_4d[:, :1]

        return mask_4d.to(self.device)

    def make_watermark_pattern(self, batch_size, out_size=None):
        """
        Generate the owner-specific reference watermark pattern.

        Args:
            batch_size: number of images in the batch.
            out_size: optional spatial output size (H, W).
                      If provided, the canonical 32x32 pattern is resized to this size.

        Returns:
            Tensor with shape [B, 3, H, W] if out_size is given, otherwise [B, 3, img_size, img_size].
        """
        self._require_freq_mask()

        mask = self._resize_real_mask_to_base_size(self.freq_mask)
        masked_fft = self.base_fft_noise * mask

        masked_fft_unshift = torch.fft.ifftshift(masked_fft, dim=(-2, -1))
        pattern_spatial = torch.fft.ifft2(masked_fft_unshift).real

        std = pattern_spatial.std(dim=(1, 2, 3), keepdim=True)
        pattern_norm = pattern_spatial / (std + 1e-8)

        final_pattern = pattern_norm.repeat(batch_size, 1, 1, 1) * self.embed_strength

        if out_size is not None and final_pattern.shape[-2:] != tuple(out_size):
            final_pattern = F.interpolate(
                final_pattern,
                size=tuple(out_size),
                mode="bilinear",
                align_corners=False,
            )

        return final_pattern

    def _get_texture_mask(self, images):
        """
        Nonlinear texture-aware spatial modulation mask.
        """
        B, C, H, W = images.shape

        gray = images.mean(dim=1, keepdim=True)
        blurred = T.GaussianBlur(kernel_size=3, sigma=1.0)(gray)

        texture = torch.abs(gray - blurred)

        t_flat = texture.view(B, -1)
        t_min = t_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        t_max = t_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)

        texture_norm = (texture - t_min) / (t_max - t_min + 1e-6)
        texture_norm = torch.pow(texture_norm, 2.0)

        mask_floor = 0.1
        mask = mask_floor + (1.0 - mask_floor) * texture_norm

        return mask.detach()

    def embed_watermark_batch(self, images):
        """
        Embed watermark into a batch of images in [0, 1].

        Args:
            images: Tensor [B, C, H, W], already in [0, 1].

        Returns:
            watermarked: Tensor [B, C, H, W]
            final_pattern: Tensor [B, C, H, W]
        """
        images = images.to(self.device)
        B, C, H, W = images.shape

        self._require_freq_mask()

        base_pattern = self.make_watermark_pattern(B, out_size=(H, W))
        t_mask = self._get_texture_mask(images)

        if t_mask.dim() == 2:
            t_mask = t_mask.unsqueeze(0).unsqueeze(0)
        elif t_mask.dim() == 3:
            t_mask = t_mask.unsqueeze(1)

        if t_mask.shape[-2:] != (H, W):
            t_mask = F.interpolate(
                t_mask,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

        if base_pattern.shape[1] != C:
            if base_pattern.shape[1] == 1:
                base_pattern = base_pattern.repeat(1, C, 1, 1)
            else:
                base_pattern = base_pattern[:, :1].repeat(1, C, 1, 1)

        final_pattern = base_pattern * t_mask
        watermarked = images + final_pattern

        return torch.clamp(watermarked, 0, 1), final_pattern

    def _require_freq_mask(self):
        if self.freq_mask is None:
            raise RuntimeError("Load the checkpoint's freq_mask.pth before evaluation.")

    def load_freq_mask(self, path):
        mask = torch.load(path, map_location=self.device, weights_only=True)
        if not isinstance(mask, torch.Tensor) or mask.ndim not in (2, 3, 4):
            raise ValueError("Expected a 2D, 3D or 4D frequency-mask tensor.")
        if not torch.isfinite(mask).all():
            raise ValueError("Frequency mask contains non-finite values.")
        self.freq_mask = mask.to(self.device)

