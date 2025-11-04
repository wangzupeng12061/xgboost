#!/usr/bin/env python3
"""
测试行业数据缓存加载
"""

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import time
from src.data.data_loader import DataLoader
import yaml

# 加载配置
with open("config/config.yaml", 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 测试从缓存加载
print("=" * 60)
print("测试行业数据缓存加载")
print("=" * 60)

print("\n1. 使用缓存加载:")
start = time.time()
loader = DataLoader(
    source="tushare",
    token=config['data']['token'],
    use_cache=True,
    cache_dir="./data"
)

industry_df = loader.load_industry_data()
elapsed = time.time() - start

print(f"  ✓ 加载完成")
print(f"  记录数: {len(industry_df)}")
print(f"  列: {', '.join(industry_df.columns)}")
print(f"  耗时: {elapsed:.3f}秒")

# 测试筛选特定股票
print("\n2. 加载指定股票的行业数据:")
test_codes = ['000001.SZ', '600000.SH', '000002.SZ']
start = time.time()
industry_filtered = loader.load_industry_data(stock_codes=test_codes)
elapsed = time.time() - start

print(f"  ✓ 加载完成")
print(f"  记录数: {len(industry_filtered)}")
print(f"  耗时: {elapsed:.3f}秒")
print(f"\n  示例数据:")
print(industry_filtered.to_string(index=False))

print("\n" + "=" * 60)
print("✓ 行业数据缓存功能正常!")
print("=" * 60)
