import numpy as np
import math
from scipy.fftpack import fft2, fftshift
import torch
from sklearn.cluster import KMeans
from PIL import Image

class FrequencyAnalyzer:
    def __init__(self, img_size: int = 32, device: str = "cuda"):
        self.img_size = int(img_size)
        self.h, self.w = self.img_size, self.img_size
        self.device = device
        self._validate_device()
    def _validate_device(self):
        if isinstance(self.device, torch.device):
            if self.device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA设备不可用，但指定了CUDA设备")
        elif isinstance(self.device, str):
            if self.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA设备不可用，但指定了CUDA设备")
            self.device = torch.device(self.device if self.device != "cuda" else "cuda:0")
        else:
            raise TypeError("device 必须是 torch.device 或 字符串")
    def grad_fft_analysis(self, model, dataset, num_samples: int = 200):
        """
        基于梯度 → 频域的敏感度分析：
          1) 对输入图像 x 求 dL/dx（对当前预测类别的负 logit）
          2) 对梯度在每个通道上做 2D FFT，取幅度谱
          3) 通道求平均后累加到 heatmap
        返回归一化到 [0,1] 的热力图。
        """
        model = model.to(self.device)
        model.eval()

        if len(dataset) == 0:
            print("警告: 数据集为空，返回默认热力图")
            return self._create_default_heatmap()

        num_samples = min(int(num_samples), len(dataset))
        indices = np.random.choice(len(dataset), size=num_samples, replace=False)

        heatmap = np.zeros((self.h, self.w), dtype=np.float64)
        used = 0

        print("==================================================")
        print("开始 Grad-FFT 频率敏感度分析")
        print("==================================================")
        print(f"Grad-FFT: 使用 {num_samples} 个样本")

        for k, idx in enumerate(indices):
            sample = dataset[idx]
            if isinstance(sample, (tuple, list)):
                img = sample[0]
            else:
                img = sample

            if not isinstance(img, torch.Tensor):
                img_np = np.array(img)
                if img_np.ndim == 2:
                    img_np = img_np[None, :, :]
                elif img_np.ndim == 3:
                    img_np = np.transpose(img_np, (2, 0, 1))
                else:
                    raise ValueError(f"不支持的图像形状: {img_np.shape}")
                img = torch.from_numpy(img_np).float() / 255.0

            img = img.to(self.device)
            img = img.unsqueeze(0)
            img.requires_grad_(True)

            model.zero_grad()
            out = model(img)
            pred = out.argmax(dim=1)
            loss = -out[0, pred.item()]
            loss.backward()

            grad = img.grad.detach()[0]
            grad_np = grad.cpu().numpy()

            if grad_np.ndim == 3:
                mags = []
                for c in range(grad_np.shape[0]):
                    G = fft2(grad_np[c])
                    G_shift = fftshift(G)
                    mags.append(np.abs(G_shift))
                mag = np.mean(np.stack(mags, axis=0), axis=0)
            else:
                G = fft2(grad_np)
                G_shift = fftshift(G)
                mag = np.abs(G_shift)

            if mag.shape != (self.h, self.w):
                mag = np.array(Image.fromarray(mag).resize((self.w, self.h)))

            heatmap += mag.astype(np.float64)
            used += 1

            if (k + 1) % 20 == 0:
                print(f"处理样本 {k + 1}/{num_samples}")

        if used == 0:
            print("Grad-FFT: 无有效样本，返回默认热力图")
            return self._create_default_heatmap()

        heatmap /= (heatmap.max() + 1e-8)
        print("\nGrad-FFT 统计:")
        print(f"  有效样本数: {used}")
        print(f"  热力图范围: [{float(heatmap.min()):.3f}, {float(heatmap.max()):.3f}]")

        return heatmap.astype(np.float32)
    def _create_default_heatmap(self):
        """
        当频率分析失败或数据不足时，创建一个默认的“中低频更敏感”的热力图。
        """
        heatmap = np.zeros((self.h, self.w), dtype=np.float32)
        center_i, center_j = self.h // 2, self.w // 2
        max_r = float(center_i) * 0.8

        for i in range(self.h):
            for j in range(self.w):
                dist = math.sqrt((i - center_i) ** 2 + (j - center_j) ** 2)
                if dist <= max_r:
                    heatmap[i, j] = 1.0 - (dist / max_r) * 0.5

        return heatmap
    def _get_default_mid_low_frequencies(self):
        """
        当没有明显敏感点时，返回一个“中低频环形区域”的位置列表。
        """
        center_i, center_j = self.h // 2, self.w // 2
        inner_r = 3
        outer_r = center_i * 0.6

        positions = []
        for i in range(self.h):
            for j in range(self.w):
                dist = math.sqrt((i - center_i) ** 2 + (j - center_j) ** 2)
                if inner_r <= dist <= outer_r:
                    positions.append((i, j))

        if len(positions) < 50:
            positions = []
            for i in range(center_i - 8, center_i + 9):
                for j in range(center_j - 8, center_j + 9):
                    if 0 <= i < self.h and 0 <= j < self.w:
                        positions.append((i, j))

        print(f"使用默认中低频区域: {len(positions)} 个位置")
        return positions
    def frequency_sensitivity_clustering(self, heatmap, rho: float = 0.3):
        """
        频率敏感度聚类：
          - 先在半径 radius 内，选出热力图值 >= rho 的位置；
          - 若无，则退回默认中低频区域；
          - 否则使用 KMeans(n_clusters=2) 对 (稳定性, 距离) 做聚类，
            选择平均距离中心更近的簇。
        """
        center_i, center_j = self.h // 2, self.w // 2
        radius = min(center_i, center_j) - 2
        print(f"频率敏感度聚类: 阈值ρ={rho}, 分析半径={radius}")

        sensitivity_map = np.zeros((self.h, self.w), dtype=np.uint8)
        positions = []

        for i in range(self.h):
            for j in range(self.w):
                dist = math.sqrt((i - center_i) ** 2 + (j - center_j) ** 2)
                if dist <= radius and heatmap[i, j] >= rho:
                    sensitivity_map[i, j] = 1
                    positions.append((i, j))

        print(f"  敏感频率点数量: {len(positions)}")
        if len(positions) == 0:
            print("  敏感点数量不足，切换到默认中低频区域")
            return self._get_default_mid_low_frequencies()

        features = []
        for (i, j) in positions:
            dist = math.sqrt((i - center_i) ** 2 + (j - center_j) ** 2)
            stability = float(heatmap[i, j])
            features.append([stability, dist])
        features = np.array(features, dtype=np.float32)

        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + 1e-6
        features_norm = (features - mean) / std

        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features_norm)

        cluster_info = []
        for cid in range(2):
            mask = clusters == cid
            if not np.any(mask):
                continue
            cluster_features = features[mask]
            cluster_info.append(
                {
                    "id": cid,
                    "size": int(mask.sum()),
                    "avg_distance": float(cluster_features[:, 1].mean()),
                    "avg_stability": float(cluster_features[:, 0].mean()),
                }
            )

        if not cluster_info:
            print("  无有效聚类结果，使用默认中低频区域")
            return self._get_default_mid_low_frequencies()

        cluster_info.sort(key=lambda x: x["avg_distance"])
        selected_cluster = cluster_info[0]
        selected_positions = [
            pos for pos, c in zip(positions, clusters) if c == selected_cluster["id"]
        ]

        print(
            f"  选中簇ID: {selected_cluster['id']}, "
            f"大小: {len(selected_positions)}, "
            f"平均距离: {selected_cluster['avg_distance']:.2f}, "
            f"平均稳定性: {selected_cluster['avg_stability']:.3f}"
        )

        if len(selected_positions) == 0:
            print("  聚类选中簇为空，回退到默认中低频区域")
            return self._get_default_mid_low_frequencies()

        return selected_positions
    def get_mid_low_frequency_mask(self, selected_positions):
        """
        将频率位置列表转换为二维掩码 (numpy.ndarray[H,W])。
        如果 selected_positions 为空，则使用默认中低频区域。
        """
        if not selected_positions:
            selected_positions = self._get_default_mid_low_frequencies()

        mask = np.zeros((self.h, self.w), dtype=np.float32)
        for (i, j) in selected_positions:
            if 0 <= i < self.h and 0 <= j < self.w:
                mask[i, j] = 1.0
        return mask
