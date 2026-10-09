import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


class BoundarySelector:
    """
    使用“样本损失 × (1 - 预测 margin) + 熵项”来选择边界样本。

    记：
        - logits = model(x)
        - p = softmax(logits)
        - loss_i = CE(logits_i, y_i)（逐样本交叉熵）
        - margin_i = p_i(top1) - p_i(top2)
        - entropy_i = -∑_c p_i(c) log p_i(c)

    定义边界分数:
        score_i = loss_i * (1 - margin_i) + entropy_weight * entropy_i

    分数越大，表示样本越“难”、越靠近决策边界。
    """

    def __init__(
        self,
        device=None,
        batch_size: int = 128,
        num_workers: int = 4,
        entropy_weight: float = 0.05,
        only_correct: bool = True,
        invalid_score: float = -1e9,
    ):
        """
        Args:
            device: 运行设备（例如 "cuda" 或 "cpu"），为 None 时自动检测。
            batch_size: 评估时的批大小。
            num_workers: DataLoader 的 num_workers。
            entropy_weight: 熵项的权重 λ。
            only_correct: 是否只在“当前预测正确”的样本中选边界样本。
            invalid_score: 用于标记“不可用样本”的极小分数。
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.entropy_weight = entropy_weight
        self.only_correct = only_correct
        self.invalid_score = float(invalid_score)

    @torch.no_grad()
    def select_boundary_samples(self, model, train_dataset, n_samples: int):
        """
        选择 n_samples 个最接近决策边界的样本。

        Args:
            model: 已经训练好的分类模型（你重新训练并加载权重后的 ResNet）。
            train_dataset: 用来选择边界样本的训练数据集（例如 CIFAR10）。
                要求 __getitem__ 至少返回 (image, target, ...)。
            n_samples: 需要选择的边界样本数量（在 only_correct=True 时会自动截断
                       到“预测正确的样本数”）。

        Returns:
            List[int]: 训练集中样本的索引列表（相对于 train_dataset 的下标）。
        """
        model = model.to(self.device)
        model.eval()

        dataset_size = len(train_dataset)
        if dataset_size == 0:
            print("[BoundarySelector] 警告：训练集大小为 0，返回空列表。")
            return []

        if n_samples <= 0:
            print("[BoundarySelector] n_samples <= 0，返回空列表。")
            return []

        n_samples = min(n_samples, dataset_size)

        data_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        all_scores = np.full(dataset_size, self.invalid_score, dtype=np.float32)

        total_correct = 0
        offset = 0

        print("[BoundarySelector] 开始计算每个样本的边界分数...")
        for batch in tqdm(data_loader, desc="Computing boundary scores"):
            if isinstance(batch, (list, tuple)):
                images = batch[0]
                targets = batch[1]
            else:
                raise ValueError(
                    "BoundarySelector 需要带标签的数据集，"
                    "__getitem__ 应返回 (image, target, ...)。"
                )

            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            bsz = images.size(0)
            end = offset + bsz

            logits = model(images)
            probs = F.softmax(logits, dim=1)

            preds = probs.argmax(dim=1)
            correct_mask = (preds == targets)

            per_sample_loss = F.cross_entropy(
                logits, targets, reduction="none"
            )

            top2_probs, _ = torch.topk(probs, k=2, dim=1)
            margin = top2_probs[:, 0] - top2_probs[:, 1]

            entropy = -torch.sum(
                probs * torch.log(probs + 1e-12),
                dim=1,
            )

            scores = per_sample_loss * (1.0 - margin) + self.entropy_weight * entropy

            if self.only_correct:
                scores = scores.clone()
                scores[~correct_mask] = self.invalid_score
                total_correct += correct_mask.sum().item()

            all_scores[offset:end] = scores.detach().cpu().numpy()
            offset = end

        if offset != dataset_size:
            print(
                f"[BoundarySelector] 警告：DataLoader 迭代的样本数 {offset} "
                f"与数据集大小 {dataset_size} 不一致，请检查数据集实现。"
            )

        if self.only_correct:
            if total_correct == 0:
                print(
                    "[BoundarySelector] 预测正确的样本数为 0，"
                    "无法从中选择边界样本，返回空列表。"
                )
                return []

            valid_mask = all_scores > (self.invalid_score * 0.5)
            valid_scores = all_scores[valid_mask]

            if valid_scores.size == 0:
                print(
                    "[BoundarySelector] 没有有效分数（可能全部为 invalid_score），"
                    "返回空列表。"
                )
                return []

            if n_samples > total_correct:
                print(
                    f"[BoundarySelector] 预测正确的样本只有 {total_correct} 个，"
                    f"小于要求的 {n_samples} 个，将 n_samples 自动截断为 {total_correct}。"
                )
                n_samples = total_correct

            print(
                f"[BoundarySelector] 有效样本数（预测正确）: {total_correct}，"
                f"score 统计：min={float(valid_scores.min()):.4f}, "
                f"max={float(valid_scores.max()):.4f}, "
                f"mean={float(valid_scores.mean()):.4f}"
            )
        else:
            valid_scores = all_scores
            print(
                "[BoundarySelector] 所有样本上的 score 统计："
                f"min={float(valid_scores.min()):.4f}, "
                f"max={float(valid_scores.max()):.4f}, "
                f"mean={float(valid_scores.mean()):.4f}"
            )

        sorted_indices = np.argsort(-all_scores)
        boundary_indices = sorted_indices[:n_samples].tolist()

        print(
            f"[BoundarySelector] 共评估 {dataset_size} 个样本，"
            f"选出 {len(boundary_indices)} 个边界样本（score 最大的前 {n_samples} 个）。"
        )
        return boundary_indices
