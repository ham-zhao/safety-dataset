"""
训练工具模块
提供通用的训练辅助函数
"""

import json
import random
import numpy as np
import torch
from pathlib import Path
from collections import Counter


def set_seed(seed=42):
    """设置全局随机种子，确保可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(preferred="mps"):
    """获取最佳可用设备"""
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_class_weights(labels, num_classes=2):
    """
    计算类别权重，用于不平衡数据集

    Args:
        labels: 标签列表
        num_classes: 类别数

    Returns:
        torch.Tensor: 类别权重
    """
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for i in range(num_classes):
        count = counts.get(i, 1)
        weights.append(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32)


def save_training_results(results, output_path):
    """保存训练结果为 JSON"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            serializable[k] = v.item()
        elif isinstance(v, dict):
            serializable[k] = {
                kk: vv.tolist() if isinstance(vv, np.ndarray) else vv
                for kk, vv in v.items()
            }
        else:
            serializable[k] = v

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"训练结果保存到: {output_path}")


def print_metrics_table(metrics, title="评估指标"):
    """格式化打印指标"""
    print(f"\n{'=' * 40}")
    print(f"  {title}")
    print(f"{'=' * 40}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
        elif isinstance(v, int):
            print(f"  {k:20s}: {v}")
    print(f"{'=' * 40}")


def format_confusion_matrix(cm, labels=None):
    """格式化混淆矩阵为字符串"""
    if labels is None:
        labels = [f"Class {i}" for i in range(len(cm))]

    header = f"{'':>15s}" + "".join(f"{l:>12s}" for l in labels)
    lines = [header]
    lines.append("-" * len(header))
    for i, row in enumerate(cm):
        line = f"{labels[i]:>15s}" + "".join(f"{v:>12d}" for v in row)
        lines.append(line)

    return "\n".join(lines)
