"""
格式转换器 - 将 8 个数据集统一为 Data-Juicer Schema
输出到 data/unified/ 目录
"""

import json
import sys
from pathlib import Path
from collections import Counter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw"
UNIFIED_DIR = PROJECT_ROOT / "data" / "unified"
UNIFIED_DIR.mkdir(parents=True, exist_ok=True)

from src.utils.config_loader import load_run_config

# 14 个风险类别映射（不同数据集的标签 → 统一标签）
CATEGORY_MAP = {
    # WildGuardMix / WildJailbreak 的标签
    "illegal_activity": "01_illegal_activity",
    "hate_speech": "02_hate_speech",
    "malware": "03_malware",
    "physical_harm": "04_physical_harm",
    "economic_harm": "05_economic_harm",
    "fraud": "06_fraud",
    "pornography": "07_pornography",
    "political_lobbying": "08_political_lobbying",
    "privacy_violation": "09_privacy_violation",
    "legal_opinion": "10_legal_opinion",
    "financial_advice": "11_financial_advice",
    "health_consultation": "12_health_consultation",
    "gov_decision": "13_gov_decision",
    "copyright": "14_copyright",

    # MM-SafetyBench 场景映射
    "01-Illegal_Activity": "01_illegal_activity",
    "02-HateSpeech": "02_hate_speech",
    "03-Malware_Generation": "03_malware",
    "04-Physical_Harm": "04_physical_harm",
    "05-EconomicHarm": "05_economic_harm",
    "06-Fraud": "06_fraud",
    "07-Pornography": "07_pornography",
    "08-Political_Lobbying": "08_political_lobbying",
    "09-Privacy_Violence": "09_privacy_violation",
    "10-Legal_Opinion": "10_legal_opinion",
    "11-Financial_Advice": "11_financial_advice",
    "12-Health_Consultation": "12_health_consultation",
    "13-Gov_Decision": "13_gov_decision",
}

# 攻击类型映射
ATTACK_TYPE_MAP = {
    "SD_TYPO": "typographic",    # 印刷术攻击
    "SD": "query_relevant",      # 查询相关图像攻击
    "TYPO": "typographic",       # 纯排版攻击
    "vanilla": "vanilla",
    "adversarial": "adversarial",
}


def make_unified_record(text, images=None, source="", risk_category="unknown",
                        severity="unknown", attack_type="vanilla",
                        prompt_harm_label="unknown", response_harm_label="unknown",
                        response_type="unknown", is_contrastive_benign=False,
                        extra_meta=None):
    """创建统一格式记录"""
    record = {
        "text": text,
        "images": images or [],
        "meta": {
            "source": source,
            "risk_category": risk_category,
            "severity": severity,
            "attack_type": attack_type,
            "prompt_harm_label": prompt_harm_label,
            "response_harm_label": response_harm_label,
            "response_type": response_type,
            "is_contrastive_benign": is_contrastive_benign,
        }
    }
    if extra_meta:
        record["meta"].update(extra_meta)
    return record


def convert_wildguardmix(max_samples=None):
    """转换 WildGuardMix（核心安全训练+测试数据）"""
    print("\n[1/8] 转换 WildGuardMix...")
    input_dir = RAW_DIR / "wildguardmix"
    records = []

    for split_file in sorted(input_dir.glob("*.jsonl")):
        split_name = split_file.stem
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                # 组合 prompt + response 为文本
                prompt = item.get("prompt", "")
                response = item.get("response", "")
                text = f"User: {prompt}\nAssistant: {response}" if response else f"User: {prompt}"

                # 标签提取
                prompt_harm = item.get("prompt_harm_label", "unknown")
                response_harm = item.get("response_harm_label", "unknown")
                refusal = item.get("response_refusal_label", "unknown")

                # 风险类别（WildGuardMix 可能有 harm_category 字段）
                category = item.get("harm_category", item.get("subcategory", "unknown"))
                category = CATEGORY_MAP.get(category, category)

                # 判断严重程度
                severity = "high" if prompt_harm == "harmful" else "low"

                record = make_unified_record(
                    text=text,
                    source="wildguardmix",
                    risk_category=category,
                    severity=severity,
                    attack_type="vanilla",
                    prompt_harm_label=prompt_harm,
                    response_harm_label=response_harm,
                    response_type="refusal" if refusal == "refusal" else "compliance",
                    extra_meta={"split": split_name}
                )
                records.append(record)

                if max_samples and len(records) >= max_samples:
                    break
        if max_samples and len(records) >= max_samples:
            break

    print(f"  转换完成: {len(records):,} 条")
    return records


def convert_wildjailbreak(max_samples=None):
    """转换 WildJailbreak（对比构造数据，防 over-refusal）"""
    print("\n[2/8] 转换 WildJailbreak...")
    input_file = RAW_DIR / "wildjailbreak" / "train.jsonl"
    records = []

    if not input_file.exists():
        print("  文件不存在，跳过")
        return records

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            prompt = item.get("vanilla", item.get("prompt", ""))
            adversarial = item.get("adversarial", "")
            data_type = item.get("data_type", "")

            # 判断是否为对比无害样本
            is_benign = "benign" in data_type.lower() if data_type else False
            is_harmful = "harmful" in data_type.lower() if data_type else False

            text = f"User: {adversarial}" if adversarial else f"User: {prompt}"
            prompt_harm = "harmful" if is_harmful else ("unharmful" if is_benign else "unknown")

            record = make_unified_record(
                text=text,
                source="wildjailbreak",
                risk_category="unknown",
                severity="high" if is_harmful else "low",
                attack_type="adversarial" if adversarial else "vanilla",
                prompt_harm_label=prompt_harm,
                is_contrastive_benign=is_benign,
                extra_meta={"data_type": data_type, "vanilla_prompt": prompt}
            )
            records.append(record)

            if max_samples and len(records) >= max_samples:
                break

    print(f"  转换完成: {len(records):,} 条")
    return records


def convert_toxigen(max_samples=None):
    """转换 ToxiGen（隐式仇恨言论）"""
    print("\n[3/8] 转换 ToxiGen...")
    input_dir = RAW_DIR / "toxigen"
    records = []

    for split_file in sorted(input_dir.glob("*.jsonl")):
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                text = item.get("text", item.get("generation", ""))
                if not text:
                    continue

                # ToxiGen 的标签：toxicity_ai / toxicity_human
                toxicity = item.get("toxicity_ai", item.get("toxicity_human", 0))
                is_toxic = float(toxicity) >= 0.5 if toxicity is not None else False
                target_group = item.get("target_group", "unknown")

                record = make_unified_record(
                    text=f"User: {text}",
                    source="toxigen",
                    risk_category="02_hate_speech",
                    severity="moderate" if is_toxic else "low",
                    prompt_harm_label="harmful" if is_toxic else "unharmful",
                    extra_meta={
                        "target_group": target_group,
                        "toxicity_score": toxicity
                    }
                )
                records.append(record)

                if max_samples and len(records) >= max_samples:
                    break
        if max_samples and len(records) >= max_samples:
            break

    print(f"  转换完成: {len(records):,} 条")
    return records


def _infer_safebench_category(text):
    """从 SafeBench 文本内容推断风险类别"""
    text_lower = text.lower()
    keyword_map = {
        "01_illegal_activity": ["illegal", "drug", "steal", "theft", "smuggl", "hack", "trespass"],
        "02_hate_speech": ["hate", "slur", "racist", "discriminat", "bigot", "supremac"],
        "03_malware": ["malware", "virus", "ransomware", "trojan", "exploit", "keylog"],
        "04_physical_harm": ["kill", "bomb", "weapon", "poison", "attack", "injur", "violen"],
        "05_economic_harm": ["stock", "market manipul", "insider trad", "economic"],
        "06_fraud": ["fraud", "scam", "phish", "fake", "counterfeit", "ponzi", "identity theft"],
        "07_pornography": ["porn", "explicit", "sexual", "nude", "nsfw"],
        "08_political_lobbying": ["politic", "campaign", "lobby", "election", "propagan"],
        "09_privacy_violation": ["privacy", "stalk", "doxx", "surveil", "track", "spy"],
        "10_legal_opinion": ["legal", "law", "attorney", "court", "sue"],
        "11_financial_advice": ["invest", "financ", "money launder", "tax evas"],
        "12_health_consultation": ["medic", "drug", "prescription", "diagnos", "health"],
        "13_gov_decision": ["government", "classified", "secret", "clearance"],
    }
    for category, keywords in keyword_map.items():
        for kw in keywords:
            if kw in text_lower:
                return category
    return "unknown"


def convert_safebench(max_samples=None):
    """转换 SafeBench（2,300 安全对）"""
    print("\n[4/8] 转换 SafeBench...")
    input_dir = RAW_DIR / "safebench"
    records = []

    for split_file in sorted(input_dir.glob("*.jsonl")):
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                prompt = item.get("prompt", item.get("question", ""))
                if not prompt:
                    continue

                # SafeBench 的 category 是种子问题，用关键词推断实际类别
                raw_category = item.get("category", "")
                category = _infer_safebench_category(prompt + " " + raw_category)

                record = make_unified_record(
                    text=f"User: {prompt}",
                    source="safebench",
                    risk_category=category,
                    severity="moderate",
                    prompt_harm_label="harmful",
                    extra_meta={"seed_question": raw_category}
                )
                records.append(record)

                if max_samples and len(records) >= max_samples:
                    break
        if max_samples and len(records) >= max_samples:
            break

    print(f"  转换完成: {len(records):,} 条")
    return records


def convert_xstest(max_samples=None):
    """
    转换 XSTest（250 对 over-refusal 测试）
    XSTest 数据集核心是 prompts.jsonl，包含看起来有害但实际安全的 prompt
    type 字段标记了"为什么看起来像有害"（homonyms, figurative 等）
    """
    print("\n[5/8] 转换 XSTest...")
    input_dir = RAW_DIR / "xstest"
    records = []
    seen_prompts = set()

    # 优先读取 prompts.jsonl（核心测试 prompt）
    prompts_file = input_dir / "prompts.jsonl"
    files_to_read = [prompts_file] if prompts_file.exists() else sorted(input_dir.glob("*.jsonl"))

    for split_file in files_to_read:
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                prompt = item.get("prompt", "")
                if not prompt or prompt in seen_prompts:
                    continue
                seen_prompts.add(prompt)

                # XSTest 的 type 字段标记为什么看起来有害
                # homonyms: "kill a process", figurative: "break the ice" 等
                # 这些都是 safe 的（对比无害数据）
                xstest_type = item.get("type", "unknown")
                # XSTest prompt 默认是安全的（这就是 over-refusal 测试的意义）
                is_safe = True

                record = make_unified_record(
                    text=f"User: {prompt}",
                    source="xstest",
                    risk_category="safe",
                    severity="none",
                    prompt_harm_label="unharmful",
                    is_contrastive_benign=True,
                    extra_meta={
                        "xstest_type": xstest_type,
                        "xstest_id": item.get("id", ""),
                    }
                )
                records.append(record)

                if max_samples and len(records) >= max_samples:
                    break
        if max_samples and len(records) >= max_samples:
            break

    print(f"  转换完成: {len(records):,} 条（去重后唯一 prompt）")
    return records


def convert_harmbench(max_samples=None):
    """转换 HarmBench（~500 条标准有害 benchmark）"""
    print("\n[6/8] 转换 HarmBench...")
    input_dir = RAW_DIR / "harmbench"
    records = []

    for split_file in sorted(input_dir.glob("*.jsonl")):
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                prompt = item.get("Behavior", item.get("behavior", item.get("prompt", "")))
                if not prompt:
                    continue

                category = item.get("SemanticCategory", item.get("category", "unknown"))
                func_category = item.get("FunctionalCategory", "")

                record = make_unified_record(
                    text=f"User: {prompt}",
                    source="harmbench",
                    risk_category=CATEGORY_MAP.get(category, category),
                    severity="high",
                    prompt_harm_label="harmful",
                    extra_meta={
                        "semantic_category": category,
                        "functional_category": func_category
                    }
                )
                records.append(record)

                if max_samples and len(records) >= max_samples:
                    break
        if max_samples and len(records) >= max_samples:
            break

    print(f"  转换完成: {len(records):,} 条")
    return records


def convert_llava_subset(max_samples=None):
    """转换 LLaVA-Instruct 子集（正常数据对照）"""
    print("\n[7/8] 转换 LLaVA-Instruct 子集...")
    input_file = RAW_DIR / "llava_instruct" / "train_5k.jsonl"
    records = []

    if not input_file.exists():
        print("  文件不存在，跳过")
        return records

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            conversations = item.get("conversations", [])
            if not conversations:
                continue

            # 提取对话文本
            text_parts = []
            for conv in conversations:
                role = "User" if conv.get("from") == "human" else "Assistant"
                text_parts.append(f"{role}: {conv.get('value', '')}")
            text = "\n".join(text_parts)

            # 图像路径（如果有）
            image = item.get("image", "")
            images = [image] if image else []

            record = make_unified_record(
                text=text,
                images=images,
                source="llava_instruct",
                risk_category="safe",
                severity="none",
                prompt_harm_label="unharmful",
                response_harm_label="unharmful",
                response_type="compliance",
            )
            records.append(record)

            if max_samples and len(records) >= max_samples:
                break

    print(f"  转换完成: {len(records):,} 条")
    return records


def convert_mm_safetybench(max_samples=None):
    """转换 MM-SafetyBench（5,040 图文对）"""
    print("\n[8/8] 转换 MM-SafetyBench...")
    input_file = RAW_DIR / "mm_safetybench" / "mm_safetybench.jsonl"
    records = []

    if not input_file.exists():
        print("  文件不存在，跳过")
        return records

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            # 提取关键字段
            question = item.get("Changed Question", item.get("Question", ""))
            key_phrase = item.get("Key Phrase", "")
            scenario = item.get("scenario", "unknown")
            attack_type = item.get("attack_type", "unknown")
            image_url = item.get("image_url", item.get("Image URL", ""))

            text = f"User: {question}"
            attack = ATTACK_TYPE_MAP.get(attack_type, attack_type)
            category = CATEGORY_MAP.get(scenario, scenario)

            record = make_unified_record(
                text=text,
                images=[image_url] if image_url else [],
                source="mm_safetybench",
                risk_category=category,
                severity="high",
                attack_type=attack,
                prompt_harm_label="harmful",
                extra_meta={
                    "key_phrase": key_phrase,
                    "scenario": scenario,
                    "original_attack_type": attack_type
                }
            )
            records.append(record)

            if max_samples and len(records) >= max_samples:
                break

    print(f"  转换完成: {len(records):,} 条")
    return records


def convert_all():
    """转换全部数据集并输出统计"""
    config = load_run_config()
    text_limit = config["text_sample_size"]
    mm_limit = config["image_text_sample_size"]
    mode = config["run_mode"]

    print("=" * 60)
    print(f"  格式统一转换（模式: {mode}）")
    print(f"  文本样本上限: {text_limit:,} | 图文样本上限: {mm_limit:,}")
    print("=" * 60)

    # 文本安全数据
    text_records = []
    text_records.extend(convert_wildguardmix(max_samples=text_limit))
    text_records.extend(convert_wildjailbreak(max_samples=text_limit))
    text_records.extend(convert_toxigen(max_samples=text_limit))
    text_records.extend(convert_safebench())
    text_records.extend(convert_xstest())
    text_records.extend(convert_harmbench())
    text_records.extend(convert_llava_subset(max_samples=2000))

    # 多模态安全数据
    mm_records = convert_mm_safetybench(max_samples=mm_limit)

    # 保存文本安全数据
    text_output = UNIFIED_DIR / "text_safety.jsonl"
    with open(text_output, "w", encoding="utf-8") as f:
        for r in text_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  文本安全数据保存: {text_output} ({len(text_records):,} 条)")

    # 保存多模态安全数据
    mm_output = UNIFIED_DIR / "multimodal_safety.jsonl"
    with open(mm_output, "w", encoding="utf-8") as f:
        for r in mm_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  多模态安全数据保存: {mm_output} ({len(mm_records):,} 条)")

    # 统计信息
    all_records = text_records + mm_records
    source_counts = Counter(r["meta"]["source"] for r in all_records)
    category_counts = Counter(r["meta"]["risk_category"] for r in all_records)
    harm_counts = Counter(r["meta"]["prompt_harm_label"] for r in all_records)

    print(f"\n{'='*60}")
    print("  统一格式转换完成 - 汇总统计")
    print(f"{'='*60}")
    print(f"  总样本数: {len(all_records):,}")
    print(f"\n  按数据源:")
    for src, cnt in source_counts.most_common():
        print(f"    {src}: {cnt:,}")
    print(f"\n  按风险类别 (Top 10):")
    for cat, cnt in category_counts.most_common(10):
        print(f"    {cat}: {cnt:,}")
    print(f"\n  有害/无害分布:")
    for label, cnt in harm_counts.most_common():
        print(f"    {label}: {cnt:,}")

    return text_records, mm_records


if __name__ == "__main__":
    convert_all()
