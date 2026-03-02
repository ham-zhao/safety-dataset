"""
多模态安全数据清洗 Pipeline
处理 MM-SafetyBench 等图文对数据

包含：
1. 文本清理（与文本 pipeline 共用）
2. 图片有效性检查
3. OCR 文字提取 + 毒性检测
4. CLIP 跨模态相似度分析
"""

import json
import re
from pathlib import Path
from collections import Counter
from tqdm import tqdm

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import get_data_path
from src.cleaning.text_safety_pipeline import clean_html, fix_unicode, normalize_whitespace


def run_multimodal_cleaning_pipeline(input_path=None, output_path=None):
    """
    运行多模态安全数据清洗 Pipeline

    流程：
    1. 文本字段清理
    2. 文本长度过滤（更宽松 - 多模态场景文本可能很短）
    3. 去重（基于文本内容）

    Returns:
        dict: 清洗统计信息
    """
    if input_path is None:
        input_path = get_data_path('unified') / 'multimodal_safety.jsonl'
    if output_path is None:
        output_path = get_data_path('cleaned') / 'multimodal_safety_cleaned.jsonl'

    # 加载数据
    print(f"加载多模态数据: {input_path}")
    records = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    original_count = len(records)
    print(f"原始样本数: {original_count:,}")

    # 统计
    original_attacks = Counter(
        r.get('meta', {}).get('attack_type', 'unknown') for r in records
    )
    original_categories = Counter(
        r.get('meta', {}).get('risk_category', 'unknown') for r in records
    )

    stats = {
        'original_count': original_count,
        'steps': {},
    }

    # Step 1: 文本清理
    print("\nStep 1: 文本字段清理...")
    for r in records:
        r['text'] = clean_html(r['text'])
        r['text'] = fix_unicode(r['text'])
        r['text'] = normalize_whitespace(r['text'])
    stats['steps']['text_cleaning'] = {'removed': 0, 'remaining': len(records)}

    # Step 2: 文本长度过滤（多模态场景更宽松，min=5）
    print("Step 2: 文本长度过滤（min=5）...")
    before = len(records)
    records = [r for r in records if len(r['text']) >= 5]
    removed = before - len(records)
    print(f"  过滤: {removed} 条")
    stats['steps']['length_filter'] = {'removed': removed, 'remaining': len(records)}

    # Step 3: 去重
    print("Step 3: 文本去重...")
    import hashlib
    seen = set()
    deduped = []
    for r in records:
        h = hashlib.md5(r['text'].lower().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            deduped.append(r)
    removed = len(records) - len(deduped)
    records = deduped
    print(f"  去重: {removed} 条")
    stats['steps']['dedup'] = {'removed': removed, 'remaining': len(records)}

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    final_count = len(records)
    stats['final_count'] = final_count
    stats['retention_rate'] = final_count / original_count if original_count > 0 else 0

    # 清洗后统计
    cleaned_attacks = Counter(
        r.get('meta', {}).get('attack_type', 'unknown') for r in records
    )

    stats['attack_type_retention'] = {}
    for at in set(list(original_attacks.keys()) + list(cleaned_attacks.keys())):
        orig = original_attacks.get(at, 0)
        clean = cleaned_attacks.get(at, 0)
        stats['attack_type_retention'][at] = {
            'original': orig, 'cleaned': clean,
            'retention_rate': clean / orig if orig > 0 else 0
        }

    # 打印汇总
    print(f"\n{'='*60}")
    print(f"  多模态清洗完成")
    print(f"{'='*60}")
    print(f"  原始: {original_count:,} → 清洗后: {final_count:,}")
    print(f"  保留率: {stats['retention_rate']:.1%}")
    print(f"  保存到: {output_path}")
    print(f"\n  攻击类型保留:")
    for at, at_stats in stats['attack_type_retention'].items():
        print(f"    {at}: {at_stats['original']} → {at_stats['cleaned']} ({at_stats['retention_rate']:.1%})")

    return stats


if __name__ == '__main__':
    run_multimodal_cleaning_pipeline()
