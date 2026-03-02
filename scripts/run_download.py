#!/usr/bin/env python3
"""
数据下载 + 格式转换 一键脚本
用法: python scripts/run_download.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import print_config
from src.data_download.download_all import download_all
from src.data_download.format_converter import convert_all


def main():
    print("\n" + "=" * 60)
    print("  阶段一：数据下载 + 格式转换")
    print("=" * 60)

    # 显示当前配置
    print_config()

    # Step 1: 下载数据集
    print("\n>>> Step 1: 下载数据集...")
    failed = download_all()

    # Step 2: 格式转换
    print("\n>>> Step 2: 统一格式转换...")
    text_records, mm_records = convert_all()

    print("\n" + "=" * 60)
    print("  数据下载 + 格式转换完成！")
    print(f"  文本数据: {len(text_records):,} 条")
    print(f"  多模态数据: {len(mm_records):,} 条")
    if failed:
        print(f"  ⚠ {len(failed)} 个数据集下载失败，请检查网络或代理设置")
    print("=" * 60)


if __name__ == "__main__":
    main()
