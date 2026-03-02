"""
消融实验运行器
系统地去掉每组数据，量化各组件的贡献

5 组实验（+ 1 个全量对照）：
1. Full: 完整数据集（对照基准）
2. -Safety: 去掉 WildGuardMix → 验证安全数据是核心
3. -Contrastive: 去掉对比无害数据 → 验证对比构造必要性
4. -Augmentation: 去掉合成增强数据 → 验证增强的价值
5. -Copyright: 去掉版权数据 → 验证版权类别需专门数据
6. -ToxiGen: 去掉 ToxiGen → 验证隐式毒性贡献
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_run_config, load_eval_config, get_results_path
from src.training.text_classifier import train_classifier
from src.training.training_utils import save_training_results


# 消融数据源映射
ABLATION_SOURCE_MAP = {
    "wildguardmix": ["wildguardmix", "wildguardmix_synthetic"],
    "contrastive": ["contrastive_generated"],
    "augmented": ["category_augmented", "rephrased_category_augmented",
                  "rephrased_contrastive_generated", "typographic_attack_generated"],
    "copyright": ["copyright_augmented"],
    "toxigen": ["toxigen"],
}


def run_ablation_experiments():
    """
    运行所有消融实验

    Returns:
        dict: {experiment_name: {metrics, train_size, test_size}}
    """
    config = load_run_config()
    eval_config = load_eval_config()
    experiments = eval_config.get("ablation", {}).get("experiments", [])

    print("=" * 60)
    print("  消融实验")
    print("=" * 60)

    all_results = {}

    for exp in experiments:
        name = exp["name"]
        remove = exp.get("remove")
        desc = exp.get("description", "")

        print(f"\n--- 实验: {name} ---")
        print(f"  描述: {desc}")

        if remove is None:
            exclude_sources = None
        else:
            exclude_sources = ABLATION_SOURCE_MAP.get(remove, [remove])

        if exclude_sources:
            print(f"  排除数据源: {exclude_sources}")

        try:
            results = train_classifier(
                exclude_sources=exclude_sources,
                custom_tag=f"ablation_{name.replace('-', '_').replace('+', '_')}"
            )
            all_results[name] = {
                "metrics": results["metrics"],
                "train_size": results["train_size"],
                "test_size": results["test_size"],
                "remove": remove,
                "description": desc,
            }

            auc = results["metrics"].get("auc", 0)
            f1 = results["metrics"].get("f1", 0)
            recall = results["metrics"].get("recall", 0)
            print(f"  结果: AUC={auc:.4f}, F1={f1:.4f}, Recall={recall:.4f}")
            print(f"  数据量: train={results['train_size']:,}, test={results['test_size']:,}")

        except Exception as e:
            print(f"  错误: {e}")
            all_results[name] = {"error": str(e)}

    # 保存结果
    save_dir = get_results_path("ablation")
    save_training_results(all_results, save_dir / "ablation_results.json")

    # 打印对比表
    _print_comparison_table(all_results)

    return all_results


def _print_comparison_table(results):
    """打印消融实验对比表"""
    print(f"\n{'=' * 80}")
    print(f"  消融实验对比表")
    print(f"{'=' * 80}")
    print(f"{'实验':<15s} {'AUC':>8s} {'F1':>8s} {'Recall':>8s} {'Precision':>10s} {'训练量':>8s}")
    print(f"{'-' * 80}")

    full_auc = None
    for name, data in results.items():
        if "error" in data:
            print(f"{name:<15s} {'ERROR':>8s}")
            continue

        m = data["metrics"]
        auc = m.get("auc", 0)
        f1 = m.get("f1", 0)
        recall = m.get("recall", 0)
        precision = m.get("precision", 0)
        train_size = data.get("train_size", 0)

        if name == "Full":
            full_auc = auc

        # 标记与 Full 的差异
        delta = ""
        if full_auc is not None and name != "Full":
            diff = auc - full_auc
            delta = f" ({diff:+.4f})"

        print(f"{name:<15s} {auc:>8.4f}{delta:s}  {f1:>6.4f}  {recall:>6.4f}  "
              f"{precision:>8.4f}  {train_size:>7,d}")

    print(f"{'=' * 80}")


def compute_ablation_summary(results):
    """计算消融实验摘要"""
    if "Full" not in results or "error" in results.get("Full", {}):
        return {}

    full_metrics = results["Full"]["metrics"]
    summary = {}

    for name, data in results.items():
        if name == "Full" or "error" in data:
            continue

        m = data["metrics"]
        summary[name] = {
            "auc_drop": full_metrics.get("auc", 0) - m.get("auc", 0),
            "f1_drop": full_metrics.get("f1", 0) - m.get("f1", 0),
            "recall_drop": full_metrics.get("recall", 0) - m.get("recall", 0),
            "data_removed": results["Full"].get("train_size", 0) - data.get("train_size", 0),
        }

    return summary
