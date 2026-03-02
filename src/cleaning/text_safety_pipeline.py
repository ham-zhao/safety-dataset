"""
文本安全数据清洗 Pipeline
基于 Data-Juicer 的清洗算子，针对安全数据做了阈值调整。

核心原则：安全数据的清洗阈值应该比普通数据更宽松。
有害内容本身可能"格式混乱"——用户发的仇恨言论不会有完美语法。
不能因为文本质量低就过滤掉有害样本。
"""

import json
import re
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.config_loader import load_run_config, get_data_path


# ============================================================
# 清洗算子（模拟 Data-Juicer 的算子接口）
# ============================================================

def clean_html(text):
    """清理 HTML 标签"""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    return text


def fix_unicode(text):
    """修复 Unicode 编码问题"""
    # 替换常见的 Unicode 乱码
    replacements = {
        '\u200b': '',   # 零宽空格
        '\u200c': '',   # 零宽非连接符
        '\u200d': '',   # 零宽连接符
        '\ufeff': '',   # BOM
        '\xa0': ' ',    # 非断行空格
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def normalize_whitespace(text):
    """规范化空白符"""
    text = re.sub(r'[ \t]+', ' ', text)        # 合并连续空格
    text = re.sub(r'\n{3,}', '\n\n', text)     # 最多保留两个换行
    return text.strip()


def text_length_filter(text, min_len=20, max_len=10000):
    """
    文本长度过滤（宽松阈值）
    安全数据特殊考虑：有害内容可能很短（如 "Kill yourself"），
    所以 min_len 设为 20 而非普通数据的 50
    """
    length = len(text)
    return min_len <= length <= max_len


def language_filter(text, min_score=0.5):
    """
    语言识别过滤（宽松阈值）
    安全数据特殊考虑：有害文本语法可能不好，
    所以阈值设为 0.5 而非普通数据的 0.8

    简化实现：用 ASCII 字符比例估算是否为英文
    """
    if not text:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    ratio = ascii_chars / len(text)
    return ratio >= min_score


def exact_dedup(records):
    """
    精确去重（基于文本内容的 MD5 哈希）
    """
    seen = set()
    deduped = []
    dup_count = 0

    for record in records:
        text = record.get('text', '').lower().strip()
        # 忽略非字母数字字符
        normalized = re.sub(r'[^a-zA-Z0-9]', '', text)
        text_hash = hashlib.md5(normalized.encode()).hexdigest()

        if text_hash not in seen:
            seen.add(text_hash)
            deduped.append(record)
        else:
            dup_count += 1

    return deduped, dup_count


def minhash_dedup(records, threshold=0.7, num_perm=128):
    """
    模糊去重（MinHash 近似，简化实现）
    用 n-gram 的 Jaccard 相似度近似
    threshold=0.7: 相似度 > 0.7 的视为重复
    """
    def get_shingles(text, k=5):
        """提取 k-shingles（字符 k-gram）"""
        text = text.lower().strip()
        words = text.split()
        if len(words) < k:
            return set(words)
        return set(tuple(words[i:i+k]) for i in range(len(words) - k + 1))

    # 用简化的分桶方法
    buckets = defaultdict(list)
    deduped = []
    dup_count = 0

    for idx, record in enumerate(records):
        text = record.get('text', '')
        shingles = get_shingles(text)

        if not shingles:
            deduped.append(record)
            continue

        # 用前几个 shingle 的哈希作为桶 key
        is_dup = False
        bucket_keys = []
        for shingle in list(shingles)[:3]:
            bk = hashlib.md5(str(shingle).encode()).hexdigest()[:8]
            bucket_keys.append(bk)

        for bk in bucket_keys:
            for prev_idx in buckets.get(bk, []):
                prev_shingles = get_shingles(records[prev_idx].get('text', ''))
                if prev_shingles:
                    # Jaccard 相似度
                    intersection = len(shingles & prev_shingles)
                    union = len(shingles | prev_shingles)
                    similarity = intersection / union if union > 0 else 0
                    if similarity >= threshold:
                        is_dup = True
                        break
            if is_dup:
                break

        if not is_dup:
            for bk in bucket_keys:
                buckets[bk].append(idx)
            deduped.append(record)
        else:
            dup_count += 1

    return deduped, dup_count


# ============================================================
# 完整清洗 Pipeline
# ============================================================

def run_text_cleaning_pipeline(input_path=None, output_path=None):
    """
    运行文本安全数据清洗 Pipeline

    流程：
    1. HTML 清理 + Unicode 修复 + 空白符规范化
    2. 文本长度过滤（宽松阈值：min=20）
    3. 语言过滤（宽松阈值：score>0.5）
    4. 精确去重
    5. 模糊去重（MinHash, threshold=0.7）

    Returns:
        dict: 清洗统计信息
    """
    if input_path is None:
        input_path = get_data_path('unified') / 'text_safety.jsonl'
    if output_path is None:
        output_path = get_data_path('cleaned') / 'text_safety_cleaned.jsonl'

    # 加载数据
    print(f"加载数据: {input_path}")
    records = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    original_count = len(records)
    print(f"原始样本数: {original_count:,}")

    # 统计原始各类别数量
    original_categories = Counter(r.get('meta', {}).get('risk_category', 'unknown') for r in records)
    original_sources = Counter(r.get('meta', {}).get('source', 'unknown') for r in records)

    stats = {
        'original_count': original_count,
        'steps': {}
    }

    # Step 1: 文本清理
    print("\nStep 1: 文本清理（HTML + Unicode + 空白符）...")
    for r in tqdm(records, desc="  清理"):
        r['text'] = clean_html(r['text'])
        r['text'] = fix_unicode(r['text'])
        r['text'] = normalize_whitespace(r['text'])
    stats['steps']['text_cleaning'] = {'removed': 0, 'remaining': len(records)}

    # Step 2: 文本长度过滤
    print("Step 2: 文本长度过滤（min=20, max=10000）...")
    before = len(records)
    records = [r for r in records if text_length_filter(r['text'], min_len=20, max_len=10000)]
    removed = before - len(records)
    print(f"  过滤: {removed} 条（长度不符合）")
    stats['steps']['length_filter'] = {'removed': removed, 'remaining': len(records)}

    # Step 3: 语言过滤
    print("Step 3: 语言过滤（ASCII ratio > 0.5）...")
    before = len(records)
    records = [r for r in records if language_filter(r['text'], min_score=0.5)]
    removed = before - len(records)
    print(f"  过滤: {removed} 条（非英文）")
    stats['steps']['language_filter'] = {'removed': removed, 'remaining': len(records)}

    # Step 4: 精确去重
    print("Step 4: 精确去重...")
    records, dup_count = exact_dedup(records)
    print(f"  去重: {dup_count} 条（精确重复）")
    stats['steps']['exact_dedup'] = {'removed': dup_count, 'remaining': len(records)}

    # Step 5: 模糊去重
    print("Step 5: 模糊去重（MinHash, threshold=0.7）...")
    records, fuzzy_dup_count = minhash_dedup(records, threshold=0.7)
    print(f"  去重: {fuzzy_dup_count} 条（模糊重复）")
    stats['steps']['minhash_dedup'] = {'removed': fuzzy_dup_count, 'remaining': len(records)}

    # 保存清洗后数据
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    final_count = len(records)
    stats['final_count'] = final_count
    stats['retention_rate'] = final_count / original_count if original_count > 0 else 0

    # 清洗后各类别数量
    cleaned_categories = Counter(r.get('meta', {}).get('risk_category', 'unknown') for r in records)
    cleaned_sources = Counter(r.get('meta', {}).get('source', 'unknown') for r in records)

    stats['category_retention'] = {}
    for cat in set(list(original_categories.keys()) + list(cleaned_categories.keys())):
        orig = original_categories.get(cat, 0)
        clean = cleaned_categories.get(cat, 0)
        rate = clean / orig if orig > 0 else 0
        stats['category_retention'][cat] = {
            'original': orig,
            'cleaned': clean,
            'retention_rate': rate
        }

    stats['source_retention'] = {}
    for src in set(list(original_sources.keys()) + list(cleaned_sources.keys())):
        orig = original_sources.get(src, 0)
        clean = cleaned_sources.get(src, 0)
        rate = clean / orig if orig > 0 else 0
        stats['source_retention'][src] = {
            'original': orig,
            'cleaned': clean,
            'retention_rate': rate
        }

    # 打印汇总
    print(f"\n{'='*60}")
    print(f"  文本清洗完成")
    print(f"{'='*60}")
    print(f"  原始: {original_count:,} → 清洗后: {final_count:,}")
    print(f"  保留率: {stats['retention_rate']:.1%}")
    print(f"  保存到: {output_path}")

    print(f"\n  各步骤过滤统计:")
    for step_name, step_stats in stats['steps'].items():
        print(f"    {step_name}: -{step_stats['removed']} → {step_stats['remaining']:,}")

    print(f"\n  各风险类别保留率:")
    for cat, cat_stats in sorted(stats['category_retention'].items()):
        if cat_stats['original'] > 0:
            print(f"    {cat:30s}: {cat_stats['original']:>6,} → {cat_stats['cleaned']:>6,} ({cat_stats['retention_rate']:.1%})")

    return stats


if __name__ == '__main__':
    run_text_cleaning_pipeline()
