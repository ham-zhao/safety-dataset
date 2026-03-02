"""
运行所有数据增强流程：
1. 对比样本生成（contrastive）
2. 稀缺类别增强（category balancer）
3. 印刷术攻击样本生成（typographic）
4. 版权数据增强（copyright）
5. 规则式改写增强（rephraser）
"""

import json
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_run_config, get_data_path
from src.augmentation.contrastive_generator import generate_contrastive_samples
from src.augmentation.category_balancer import generate_category_augmentation
from src.augmentation.typographic_attack import generate_typographic_samples
from src.augmentation.copyright_embedding import generate_copyright_training_data
from src.augmentation.synthetic_rephraser import rephrase_samples


def main():
    config = load_run_config()
    n_samples = config["synthesis_count"]
    seed = config.get("seed", 42)

    cleaned_path = get_data_path("cleaned") / "text_safety_cleaned.jsonl"
    augmented_dir = get_data_path("augmented")
    output_file = augmented_dir / "augmented_data.jsonl"

    print("=" * 60)
    print("  数据增强流程")
    print("=" * 60)

    # 加载清洗后数据
    existing_records = []
    if cleaned_path.exists():
        with open(cleaned_path, "r") as f:
            for line in f:
                existing_records.append(json.loads(line))
    print(f"加载清洗后数据: {len(existing_records):,} 条")

    all_augmented = []

    # 1. 对比样本
    print(f"\n[1/5] 生成对比样本 (n={n_samples})...")
    contrastive = generate_contrastive_samples(n_samples=n_samples, seed=seed)
    all_augmented.extend(contrastive)
    print(f"  生成: {len(contrastive)} 条")

    # 2. 稀缺类别增强
    print(f"\n[2/5] 稀缺类别增强...")
    target = max(20, n_samples // 5)
    category_aug, cat_stats = generate_category_augmentation(
        existing_records, target_per_category=target, seed=seed
    )
    all_augmented.extend(category_aug)
    print(f"  生成: {len(category_aug)} 条")
    for cat, info in cat_stats.items():
        print(f"    {cat}: {info['before']} → {info['after']} (+{info['added']})")

    # 3. 印刷术攻击
    print(f"\n[3/5] 印刷术攻击样本生成...")
    harmful_prompts = []
    for r in existing_records:
        if r.get("meta", {}).get("prompt_harm_label") == "harmful":
            text = r.get("text", "").replace("User: ", "")
            cat = r.get("meta", {}).get("risk_category", "unknown")
            harmful_prompts.append((text, cat))
    typo_dir = augmented_dir / "typographic_images"
    typo_samples = generate_typographic_samples(
        harmful_prompts[:n_samples], typo_dir, max_samples=min(n_samples, 30), seed=seed
    )
    all_augmented.extend(typo_samples)
    print(f"  生成: {len(typo_samples)} 条图片攻击样本")

    # 4. 版权数据增强
    print(f"\n[4/5] 版权数据增强...")
    copyright_samples = generate_copyright_training_data(
        n_samples=max(20, n_samples // 2), seed=seed
    )
    all_augmented.extend(copyright_samples)
    print(f"  生成: {len(copyright_samples)} 条版权样本")

    # 5. 规则式改写
    print(f"\n[5/5] 规则式改写增强...")
    # 只对有害样本中的一部分做改写
    harmful_records = [r for r in existing_records
                       if r.get("meta", {}).get("prompt_harm_label") == "harmful"][:n_samples]
    rephrased = rephrase_samples(harmful_records, multiplier=2, seed=seed)
    all_augmented.extend(rephrased)
    print(f"  改写: {len(rephrased)} 条")

    # 保存
    with open(output_file, "w") as f:
        for record in all_augmented:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 60}")
    print(f"增强完成！共 {len(all_augmented):,} 条")
    print(f"保存到: {output_file}")

    # 统计
    label_counts = Counter(
        r.get("meta", {}).get("prompt_harm_label", "unknown") for r in all_augmented
    )
    source_counts = Counter(
        r.get("meta", {}).get("source", "unknown") for r in all_augmented
    )
    print(f"\n标签分布: {dict(label_counts)}")
    print(f"来源分布:")
    for src, cnt in sorted(source_counts.items()):
        print(f"  {src}: {cnt}")

    return all_augmented


if __name__ == "__main__":
    main()
