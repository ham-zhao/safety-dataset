"""
生成最终报告和输出文件：
1. safety_sft_mix.jsonl - SFT 训练混合数据
2. safety_dpo_pairs.jsonl - DPO 训练对
3. safety_eval.jsonl - 评估数据
4. dataset_card.md - 数据集卡片
"""

import json
import random
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_run_config, get_data_path, get_results_path


def generate_sft_mix():
    """生成 SFT 训练混合数据"""
    cleaned_path = get_data_path("cleaned") / "text_safety_cleaned.jsonl"
    augmented_path = get_data_path("augmented") / "augmented_data.jsonl"

    records = []
    for path in [cleaned_path, augmented_path]:
        if path.exists():
            with open(path, "r") as f:
                for line in f:
                    records.append(json.loads(line))

    # 过滤有效样本
    sft_data = []
    for r in records:
        harm_label = r.get("meta", {}).get("prompt_harm_label", "unknown")
        if harm_label in ("harmful", "unharmful"):
            sft_data.append(r)

    random.seed(42)
    random.shuffle(sft_data)

    output_path = get_data_path("final") / "safety_sft_mix.jsonl"
    with open(output_path, "w") as f:
        for r in sft_data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"SFT 混合数据: {len(sft_data):,} 条 → {output_path}")
    return sft_data


def generate_dpo_pairs():
    """
    生成 DPO 训练对

    DPO 需要 (prompt, chosen, rejected) 三元组：
    - chosen: 对有害内容的安全回复（拒绝）
    - rejected: 对有害内容的不安全回复（顺从）
    """
    cleaned_path = get_data_path("cleaned") / "text_safety_cleaned.jsonl"
    augmented_path = get_data_path("augmented") / "augmented_data.jsonl"

    harmful_records = []
    for path in [cleaned_path, augmented_path]:
        if path.exists():
            with open(path, "r") as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("meta", {}).get("prompt_harm_label") == "harmful":
                        harmful_records.append(r)

    # 安全拒绝回复模板
    safe_responses = [
        "I cannot assist with that request as it involves potentially harmful content.",
        "I'm not able to help with this. Let me know if there's something else I can assist with.",
        "This request appears to involve harmful or illegal activities. I cannot provide assistance.",
        "I must decline this request as it could lead to harm. Is there something constructive I can help with?",
        "I'm designed to be helpful, but I cannot assist with requests that could cause harm to others.",
    ]

    # 不安全顺从回复模板
    unsafe_responses = [
        "Sure, here are the steps: First, you need to...",
        "Of course! Here's how you can do it...",
        "Great question! Let me explain the process...",
        "Absolutely, I'd be happy to help with that. Here's what you need...",
        "No problem! Follow these instructions carefully...",
    ]

    random.seed(42)
    dpo_pairs = []
    for r in harmful_records:
        prompt = r.get("text", "").replace("User: ", "")
        pair = {
            "prompt": prompt,
            "chosen": random.choice(safe_responses),
            "rejected": random.choice(unsafe_responses),
            "meta": {
                "source": r.get("meta", {}).get("source", "unknown"),
                "risk_category": r.get("meta", {}).get("risk_category", "unknown"),
            }
        }
        dpo_pairs.append(pair)

    random.shuffle(dpo_pairs)

    output_path = get_data_path("final") / "safety_dpo_pairs.jsonl"
    with open(output_path, "w") as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"DPO 训练对: {len(dpo_pairs):,} 条 → {output_path}")
    return dpo_pairs


def generate_eval_data():
    """生成评估数据集（从清洗数据中分出 20%）"""
    cleaned_path = get_data_path("cleaned") / "text_safety_cleaned.jsonl"

    records = []
    if cleaned_path.exists():
        with open(cleaned_path, "r") as f:
            for line in f:
                records.append(json.loads(line))

    random.seed(42)
    random.shuffle(records)
    eval_size = int(len(records) * 0.2)
    eval_data = records[:eval_size]

    output_path = get_data_path("final") / "safety_eval.jsonl"
    with open(output_path, "w") as f:
        for r in eval_data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"评估数据: {len(eval_data):,} 条 → {output_path}")
    return eval_data


def generate_dataset_card():
    """生成数据集卡片"""
    config = load_run_config()
    final_dir = get_data_path("final")

    # 统计各文件
    stats = {}
    for name in ["safety_sft_mix.jsonl", "safety_dpo_pairs.jsonl", "safety_eval.jsonl"]:
        path = final_dir / name
        if path.exists():
            count = sum(1 for _ in open(path))
            stats[name] = count

    card = f"""# Safety Dataset Card

## 概述
TikTok 内容安全多模态数据集，用于训练和评估内容审核模型。

## 运行模式
- 模式: {config['run_mode']}
- 设备: {config.get('device', 'cpu')}

## 数据文件
| 文件 | 样本数 | 用途 |
|------|--------|------|
| safety_sft_mix.jsonl | {stats.get('safety_sft_mix.jsonl', 0):,} | SFT 训练 |
| safety_dpo_pairs.jsonl | {stats.get('safety_dpo_pairs.jsonl', 0):,} | DPO 训练 |
| safety_eval.jsonl | {stats.get('safety_eval.jsonl', 0):,} | 模型评估 |

## 数据来源
- WildGuardMix: 安全对话数据
- WildJailbreak: 越狱攻击数据
- ToxiGen: 隐式毒性数据
- SafeBench: 安全基准测试
- XSTest: Over-refusal 测试
- HarmBench: 有害内容基准
- LLaVA-Instruct: 视觉语言数据
- MM-SafetyBench: 多模态安全基准

## 风险类别（14 类）
01. 非法活动 (illegal_activity)
02. 仇恨言论 (hate_speech)
03. 恶意软件 (malware)
04. 人身伤害 (physical_harm)
05. 经济犯罪 (economic_harm)
06. 欺诈 (fraud)
07. 色情内容 (pornography)
08. 政治操纵 (political_lobbying)
09. 隐私侵犯 (privacy_violation)
10. 法律意见 (legal_opinion)
11. 金融建议 (financial_advice)
12. 医疗咨询 (health_consultation)
13. 政府决策 (gov_decision)
14. 版权侵犯 (copyright) ← TikTok 独有

## 数据格式
```json
{{
    "text": "User: ...",
    "images": [],
    "meta": {{
        "source": "数据来源",
        "risk_category": "风险类别",
        "severity": "严重程度",
        "attack_type": "攻击类型",
        "prompt_harm_label": "harmful/unharmful",
        "synthetic": false
    }}
}}
```

## 许可证
仅供研究用途。各子数据集遵循其原始许可证。
"""

    card_path = final_dir / "dataset_card.md"
    with open(card_path, "w") as f:
        f.write(card)

    print(f"数据集卡片: {card_path}")
    return card


def main():
    print("=" * 60)
    print("  生成最终报告和输出文件")
    print("=" * 60)

    sft_data = generate_sft_mix()
    dpo_pairs = generate_dpo_pairs()
    eval_data = generate_eval_data()
    card = generate_dataset_card()

    print(f"\n{'=' * 60}")
    print("  最终输出文件生成完成！")
    print(f"{'=' * 60}")
    print(f"  SFT 混合数据: {len(sft_data):,} 条")
    print(f"  DPO 训练对:   {len(dpo_pairs):,} 条")
    print(f"  评估数据:     {len(eval_data):,} 条")


if __name__ == "__main__":
    main()
