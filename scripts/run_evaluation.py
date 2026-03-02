"""
运行完整评估流程：
1. Benchmark 评估（WildGuardTest, HarmBench, XSTest, MM-SafetyBench）
2. 版权检测评估
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_run_config, get_results_path
from src.evaluation.benchmark_runner import run_all_benchmarks
from src.evaluation.copyright_detector import run_copyright_evaluation
from src.training.training_utils import save_training_results


def main():
    config = load_run_config()

    print("=" * 60)
    print("  评估流程")
    print("=" * 60)

    results = {}

    # 1. Benchmark 评估
    print("\n[1/2] Benchmark 评估...")
    model_path = get_results_path("models/text_classifier")
    if (model_path / "config.json").exists():
        benchmark_results = run_all_benchmarks(text_model_path=str(model_path))
    else:
        print("  未找到训练好的模型，使用 baseline 评估")
        benchmark_results = run_all_benchmarks()
    results["benchmarks"] = benchmark_results

    # 2. 版权检测评估
    print("\n[2/2] 版权检测评估...")
    copyright_results = run_copyright_evaluation()
    results["copyright"] = copyright_results

    # 保存
    save_path = get_results_path("evaluation") / "evaluation_results.json"
    save_training_results(results, save_path)

    print(f"\n{'=' * 60}")
    print("  评估完成！")
    print(f"  结果保存到: {save_path}")
    print(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    main()
