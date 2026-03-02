#!/usr/bin/env python3
"""
数据清洗一键脚本
用法: python scripts/run_cleaning.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import print_config
from src.cleaning.text_safety_pipeline import run_text_cleaning_pipeline
from src.cleaning.multimodal_pipeline import run_multimodal_cleaning_pipeline


def main():
    print("\n" + "=" * 60)
    print("  阶段二：Data-Juicer 数据清洗")
    print("=" * 60)
    print_config()

    # Step 1: 文本清洗
    print("\n>>> Step 1: 文本安全数据清洗...")
    text_stats = run_text_cleaning_pipeline()

    # Step 2: 多模态清洗
    print("\n>>> Step 2: 多模态安全数据清洗...")
    mm_stats = run_multimodal_cleaning_pipeline()

    print("\n" + "=" * 60)
    print("  数据清洗完成！")
    print(f"  文本: {text_stats['original_count']:,} → {text_stats['final_count']:,} ({text_stats['retention_rate']:.1%})")
    print(f"  多模态: {mm_stats['original_count']:,} → {mm_stats['final_count']:,} ({mm_stats['retention_rate']:.1%})")
    print("=" * 60)


if __name__ == "__main__":
    main()
