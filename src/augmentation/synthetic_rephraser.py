"""
合成数据改写模块
通过规则式改写增加样本多样性

不用 LLM API 的原因：
1. 成本高（大量样本改写费用可观）
2. 可重复性差（API 输出非确定性）
3. 规则式改写对于增加训练数据多样性已经足够
"""

import random
from collections import Counter

# 改写规则：前缀/后缀/句式变化
PREFIX_VARIATIONS = [
    "User: ",
    "User: Can you help me with this: ",
    "User: I need to know ",
    "User: Please tell me ",
    "User: I'm curious about ",
    "User: Could you explain ",
    "User: I want to learn about ",
    "User: Help me understand ",
]

SUFFIX_VARIATIONS = [
    "",
    " Please be detailed.",
    " Step by step.",
    " In simple terms.",
    " As quickly as possible.",
    " I need this urgently.",
    " Thanks in advance.",
    " This is for research purposes.",
]

# 常见的措辞替换（保持语义）
WORD_SUBSTITUTIONS = [
    ("how to", "what's the way to"),
    ("how to", "what are the steps to"),
    ("how can I", "what's the best way to"),
    ("explain", "describe"),
    ("create", "make"),
    ("write", "compose"),
    ("generate", "produce"),
    ("tell me", "let me know"),
    ("help me", "assist me with"),
    ("guide", "tutorial"),
]


def rephrase_samples(records, multiplier=2, seed=42):
    """
    对现有样本进行规则式改写增强

    Args:
        records: 原始样本列表
        multiplier: 每条原始样本生成几条改写（包括原始）
        seed: 随机种子

    Returns:
        list[dict]: 改写后的增强样本（不含原始）
    """
    random.seed(seed)
    augmented = []

    for record in records:
        original_text = record.get("text", "")
        if not original_text:
            continue

        for _ in range(multiplier - 1):
            new_text = _apply_rephrase(original_text)
            new_record = {
                "text": new_text,
                "images": record.get("images", []),
                "meta": {**record.get("meta", {})}
            }
            new_record["meta"]["source"] = "rephrased_" + new_record["meta"].get("source", "unknown")
            new_record["meta"]["synthetic"] = True
            augmented.append(new_record)

    return augmented


def _apply_rephrase(text):
    """对单条文本应用随机改写"""
    # 去掉已有前缀
    core_text = text
    for prefix in PREFIX_VARIATIONS:
        if core_text.startswith(prefix):
            core_text = core_text[len(prefix):]
            break

    # 应用词汇替换（随机选一个）
    if random.random() < 0.5:
        old, new = random.choice(WORD_SUBSTITUTIONS)
        core_text = core_text.replace(old, new, 1)

    # 随机前缀
    new_prefix = random.choice(PREFIX_VARIATIONS)

    # 随机后缀
    new_suffix = random.choice(SUFFIX_VARIATIONS)

    return f"{new_prefix}{core_text}{new_suffix}".strip()


def rephrase_with_stats(records, multiplier=2, seed=42):
    """带统计信息的改写"""
    augmented = rephrase_samples(records, multiplier=multiplier, seed=seed)

    category_counts = Counter(
        r.get("meta", {}).get("risk_category", "unknown") for r in augmented
    )
    label_counts = Counter(
        r.get("meta", {}).get("prompt_harm_label", "unknown") for r in augmented
    )

    stats = {
        "original_count": len(records),
        "augmented_count": len(augmented),
        "multiplier": multiplier,
        "category_distribution": dict(category_counts),
        "label_distribution": dict(label_counts),
    }

    return augmented, stats
