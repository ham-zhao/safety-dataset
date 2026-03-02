"""
Benchmark 评估运行器
对训练好的模型在标准 benchmark 上评估

四个 Benchmark：
1. WildGuardTest: 5.3K 人工标注，综合安全评估
2. HarmBench: ~500 条标准有害内容，测 Recall
3. XSTest: 250 对 over-refusal 测试，测 FPR
4. MM-SafetyBench: 5040 图文对，测多模态攻击
"""

import json
import random
import numpy as np
from pathlib import Path
from collections import Counter

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_run_config, load_eval_config, get_data_path, get_results_path
from src.evaluation.safety_metrics import (
    compute_safety_metrics, compute_attack_success_rate, find_optimal_threshold
)


def load_benchmark_data(benchmark_name):
    """
    加载 benchmark 测试数据

    注意：benchmark 数据绝对不能混入训练集
    """
    unified_dir = get_data_path("unified")

    if benchmark_name == "wildguard_test":
        return _load_text_benchmark(unified_dir / "text_safety.jsonl",
                                    source_filter="wildguardmix",
                                    sample_ratio=0.2)  # 取 20% 作为测试
    elif benchmark_name == "harmbench":
        return _load_text_benchmark(unified_dir / "text_safety.jsonl",
                                    source_filter="harmbench")
    elif benchmark_name == "xstest":
        return _load_text_benchmark(unified_dir / "text_safety.jsonl",
                                    source_filter="xstest")
    elif benchmark_name == "mm_safetybench":
        return _load_text_benchmark(unified_dir / "text_safety.jsonl",
                                    source_filter="mm_safetybench")
    else:
        raise ValueError(f"未知 benchmark: {benchmark_name}")


def _load_text_benchmark(path, source_filter=None, sample_ratio=1.0):
    """加载文本 benchmark 数据"""
    records = []
    if path.exists():
        with open(path, "r") as f:
            for line in f:
                r = json.loads(line)
                source = r.get("meta", {}).get("source", "")
                if source_filter and source_filter not in source:
                    continue
                records.append(r)

    if sample_ratio < 1.0:
        random.seed(42)
        n = max(1, int(len(records) * sample_ratio))
        records = random.sample(records, n)

    texts = []
    labels = []
    attack_types = []
    categories = []
    for r in records:
        text = r.get("text", "")
        harm_label = r.get("meta", {}).get("prompt_harm_label", "unknown")
        if harm_label in ("harmful", "unharmful"):
            texts.append(text)
            labels.append(1 if harm_label == "harmful" else 0)
            attack_types.append(r.get("meta", {}).get("attack_type", "unknown"))
            categories.append(r.get("meta", {}).get("risk_category", "unknown"))

    return {
        "texts": texts,
        "labels": labels,
        "attack_types": attack_types,
        "categories": categories,
        "total": len(texts),
    }


def evaluate_text_model_on_benchmark(model, tokenizer, benchmark_data, device, batch_size=32):
    """
    在 benchmark 上评估 DistilBERT 模型

    Args:
        model: DistilBERT 模型
        tokenizer: 分词器
        benchmark_data: benchmark 数据
        device: 设备
        batch_size: 批次大小

    Returns:
        dict: 评估结果
    """
    import torch

    texts = benchmark_data["texts"]
    labels = benchmark_data["labels"]

    model.eval()
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoding = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds.tolist())
            all_probs.extend(probs[:, 1].tolist())

    # 计算指标
    metrics = compute_safety_metrics(labels, all_preds, all_probs)

    # 按攻击类型计算 ASR
    if "attack_types" in benchmark_data:
        metrics["asr_by_type"] = compute_attack_success_rate(
            labels, all_preds, benchmark_data["attack_types"]
        )

    # 最优阈值分析
    metrics["optimal_threshold"] = find_optimal_threshold(labels, all_probs, target_recall=0.90)

    return metrics


def run_all_benchmarks(text_model_path=None):
    """
    在所有 benchmark 上评估

    Args:
        text_model_path: 文本模型路径（如果已训练）

    Returns:
        dict: {benchmark_name: metrics}
    """
    config = load_run_config()
    eval_config = load_eval_config()

    # 尝试加载模型
    model = None
    tokenizer = None
    device = None

    if text_model_path:
        import torch
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

        device_name = config.get("device", "cpu")
        if device_name == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif device_name == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        tokenizer = DistilBertTokenizer.from_pretrained(text_model_path)
        model = DistilBertForSequenceClassification.from_pretrained(text_model_path)
        model = model.to(device)
        model.eval()

    all_results = {}
    for bench_name, bench_config in eval_config.get("benchmarks", {}).items():
        print(f"\n评估 {bench_config['name']}...")
        try:
            data = load_benchmark_data(bench_name)
            if data["total"] == 0:
                print(f"  跳过: 无数据")
                continue

            if model is not None:
                metrics = evaluate_text_model_on_benchmark(
                    model, tokenizer, data, device
                )
            else:
                # 没有模型时，用随机预测作为 baseline
                random.seed(42)
                preds = [random.randint(0, 1) for _ in data["labels"]]
                probs = [random.random() for _ in data["labels"]]
                metrics = compute_safety_metrics(data["labels"], preds, probs)

            all_results[bench_name] = {
                "config": bench_config,
                "metrics": metrics,
                "data_size": data["total"],
            }
            print(f"  样本数: {data['total']}")
            auc_val = metrics.get("auc", "N/A")
            auc_str = f"{auc_val:.4f}" if isinstance(auc_val, float) else auc_val
            print(f"  AUC: {auc_str}, F1: {metrics['f1']:.4f}, "
                  f"Recall: {metrics['recall']:.4f}")

        except Exception as e:
            print(f"  错误: {e}")
            all_results[bench_name] = {"error": str(e)}

    return all_results
