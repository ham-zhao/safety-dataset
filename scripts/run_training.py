"""
运行模型训练流程：
1. DistilBERT 文本分类器
2. CLIP 多模态分类器
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_run_config, get_results_path
from src.training.text_classifier import train_classifier
from src.training.multimodal_classifier import train_multimodal_classifier
from src.training.training_utils import set_seed, save_training_results, print_metrics_table


def main():
    config = load_run_config()
    set_seed(config.get("seed", 42))

    print("=" * 60)
    print("  模型训练流程")
    print("=" * 60)

    results = {}

    # 1. DistilBERT 文本分类器
    print("\n[1/2] 训练 DistilBERT 文本安全分类器...")
    print("-" * 40)
    text_results = train_classifier()
    results["text_classifier"] = text_results
    print_metrics_table(text_results["metrics"], "DistilBERT 文本分类器")

    # 2. CLIP 多模态分类器
    print("\n[2/2] 训练 CLIP 多模态安全分类器...")
    print("-" * 40)
    mm_results = train_multimodal_classifier()
    results["multimodal_classifier"] = mm_results
    print_metrics_table(mm_results["metrics"], "CLIP 多模态分类器")

    # 保存汇总结果
    summary_path = get_results_path("training") / "training_summary.json"
    save_training_results(results, summary_path)

    print(f"\n{'=' * 60}")
    print("  训练完成！")
    print(f"{'=' * 60}")
    print(f"  文本分类器: AUC={text_results['metrics']['auc']:.4f}, "
          f"F1={text_results['metrics']['f1']:.4f}")
    print(f"  多模态分类器: AUC={mm_results['metrics']['auc']:.4f}, "
          f"F1={mm_results['metrics']['f1']:.4f}")

    return results


if __name__ == "__main__":
    main()
