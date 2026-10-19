import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


DATASET_CFG = {
    "mnist": {"classes": 10, "mean": [0.1307] * 3, "std": [0.3081] * 3},
    "cifar10": {"classes": 10, "mean": [0.4914, 0.4822, 0.4465], "std": [0.2470, 0.2430, 0.2610]},
    "cifar100": {"classes": 100, "mean": [0.5071, 0.4867, 0.4408], "std": [0.2675, 0.2565, 0.2761]},
}
