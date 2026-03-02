"""
运行消融实验
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.ablation_runner import run_ablation_experiments, compute_ablation_summary
from src.training.training_utils import save_training_results
from src.utils.config_loader import get_results_path


def main():
    print("=" * 60)
    print("  消融实验")
    print("=" * 60)

    results = run_ablation_experiments()
    summary = compute_ablation_summary(results)

    if summary:
        print(f"\n消融影响排序（按 AUC 下降）:")
        sorted_ablations = sorted(summary.items(), key=lambda x: x[1]["auc_drop"], reverse=True)
        for name, s in sorted_ablations:
            print(f"  {name}: AUC -{s['auc_drop']:.4f}, F1 -{s['f1_drop']:.4f}, "
                  f"去除 {s['data_removed']:,} 条")

    save_path = get_results_path("ablation") / "ablation_summary.json"
    save_training_results({"results": results, "summary": summary}, save_path)

    return results


if __name__ == "__main__":
    main()
