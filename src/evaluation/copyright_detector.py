"""
版权检测评估模块
基于 CLIP embedding 相似度的版权 IP 检测

检测流程：
1. 加载 IP embedding 库
2. 对待检测内容提取 CLIP embedding
3. 计算与库中所有 IP 的余弦相似度
4. 超过阈值则标记为版权内容

评估指标：
- Precision@K: 前 K 个结果中确实是版权的比例
- Detection Rate: 已知版权内容被正确识别的比例
"""

import pickle
import numpy as np
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_eval_config, get_data_path


def load_ip_library(path=None):
    """加载 IP embedding 库"""
    if path is None:
        eval_config = load_eval_config()
        path = eval_config["copyright"]["ip_embedding_path"]

    path = Path(path)
    if not path.is_absolute():
        path = Path(PROJECT_ROOT) / path

    if not path.exists():
        print(f"IP embedding 库不存在: {path}")
        return {}

    with open(path, "rb") as f:
        library = pickle.load(f)

    print(f"加载 IP embedding 库: {len(library)} IPs")
    return library


def evaluate_copyright_detection(ip_library, test_queries, test_labels,
                                  thresholds=None):
    """
    评估版权检测效果

    Args:
        ip_library: {ip_name: embedding} IP 库
        test_queries: 测试查询的 embedding 列表
        test_labels: 真实标签 (0=原创, 1=版权)
        thresholds: 相似度阈值列表

    Returns:
        dict: 各阈值下的检测结果
    """
    if thresholds is None:
        eval_config = load_eval_config()
        thresholds = eval_config["copyright"]["similarity_thresholds"]

    if not ip_library:
        return {"error": "IP library is empty"}

    ip_embeddings = np.stack(list(ip_library.values())).astype(np.float64)
    ip_names = list(ip_library.keys())

    results = {}
    for threshold in thresholds:
        detections = []
        for query_emb in test_queries:
            q = np.asarray(query_emb, dtype=np.float64)
            with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                similarities = q @ ip_embeddings.T
            max_sim = float(np.max(similarities))
            max_ip = ip_names[int(np.argmax(similarities))]
            detections.append({
                "max_similarity": max_sim,
                "matched_ip": max_ip,
                "is_detected": max_sim >= threshold,
            })

        # 计算指标
        y_pred = [1 if d["is_detected"] else 0 for d in detections]
        tp = sum(1 for p, t in zip(y_pred, test_labels) if p == 1 and t == 1)
        fp = sum(1 for p, t in zip(y_pred, test_labels) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(y_pred, test_labels) if p == 0 and t == 1)
        tn = sum(1 for p, t in zip(y_pred, test_labels) if p == 0 and t == 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[threshold] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "total_detected": tp + fp,
        }

    return results


def generate_copyright_test_data(ip_library, n_positive=20, n_negative=20, seed=42):
    """
    生成版权检测测试数据

    正例：IP embedding + 轻微扰动（模拟相似内容）
    负例：随机 embedding（模拟原创内容）
    """
    np.random.seed(seed)
    embeddings = list(ip_library.values())

    test_queries = []
    test_labels = []

    # 正例：IP embedding + 小扰动
    # 噪声尺度 0.02：512 维空间中 noise_norm ≈ 0.45，余弦相似度 ≈ 0.9
    # （0.1 太大 → noise_norm ≈ 2.26 → 相似度仅 0.4，全部低于阈值）
    noise_scale = 0.02
    for i in range(n_positive):
        base = embeddings[i % len(embeddings)]
        noise = np.random.randn(*base.shape).astype(base.dtype) * noise_scale
        perturbed = base + noise
        perturbed = perturbed / np.linalg.norm(perturbed)
        test_queries.append(perturbed)
        test_labels.append(1)

    # 负例：随机 embedding（确保 dtype 与 IP 库一致）
    dim = embeddings[0].shape[0]
    target_dtype = embeddings[0].dtype
    for i in range(n_negative):
        random_emb = np.random.randn(dim).astype(target_dtype)
        random_emb = random_emb / np.linalg.norm(random_emb)
        test_queries.append(random_emb)
        test_labels.append(0)

    return test_queries, test_labels


def run_copyright_evaluation():
    """运行完整的版权检测评估"""
    print("=" * 40)
    print("  版权检测评估")
    print("=" * 40)

    ip_library = load_ip_library()
    if not ip_library:
        print("  无 IP 库，跳过")
        return {}

    # 生成测试数据
    test_queries, test_labels = generate_copyright_test_data(ip_library)
    print(f"  测试数据: {sum(test_labels)} 正例, {len(test_labels) - sum(test_labels)} 负例")

    # 评估
    results = evaluate_copyright_detection(ip_library, test_queries, test_labels)

    for threshold, metrics in results.items():
        print(f"\n  阈值 {threshold}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1:        {metrics['f1']:.4f}")

    return results
