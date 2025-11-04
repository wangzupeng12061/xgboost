#!/usr/bin/env python3
"""
下载并缓存行业分类数据
"""

import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
from pathlib import Path
import time
import yaml
from src.data.data_cache import DataCache
from src.data.data_loader import DataLoader


def download_industry_data():
    """下载行业分类数据到缓存"""
    
    print("=" * 60)
    print("行业分类数据下载工具")
    print("=" * 60)
    
    # 加载配置
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        print(f"✗ 配置文件不存在: {config_path}")
        return 1
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 初始化缓存
    cache_dir = "./data"
    cache = DataCache(cache_dir=cache_dir, expire_days=0)
    
    print(f"\n缓存目录: {cache_dir}")
    print(f"过期策略: 永不过期")
    
    # 检查是否已存在
    existing_data = cache.get_industry_data()
    if existing_data is not None:
        print(f"\n⚠️  缓存中已存在行业数据: {len(existing_data)} 条记录")
        choice = input("是否重新下载？(y/n): ").strip().lower()
        if choice != 'y':
            print("已取消")
            return
    
    # 初始化数据加载器
    print("\n初始化数据加载器...")
    loader = DataLoader(
        source="tushare",
        token=config.get('data', {}).get('token'),
        use_cache=False  # 不使用缓存，直接从API获取
    )
    
    print("\n开始下载行业分类数据...")
    start_time = time.time()
    
    try:
        # 获取所有股票的行业信息
        df = loader.load_industry_data(stock_codes=None)
        
        if df is not None and len(df) > 0:
            # 还原为ts_code列名
            if 'stock_code' in df.columns:
                df = df.rename(columns={'stock_code': 'ts_code'})
            
            # 保存到缓存
            cache.save_industry_data(df)
            
            elapsed = time.time() - start_time
            
            print(f"\n✓ 下载完成!")
            print(f"  记录数: {len(df)}")
            print(f"  列: {', '.join(df.columns)}")
            print(f"  耗时: {elapsed:.1f}秒")
            
            # 显示行业统计
            if 'industry' in df.columns:
                print(f"\n行业分布:")
                industry_counts = df['industry'].value_counts()
                print(f"  共 {len(industry_counts)} 个行业")
                print(f"  前10个行业:")
                for industry, count in industry_counts.head(10).items():
                    print(f"    {industry}: {count} 家公司")
            
            # 显示缓存文件信息
            cache_file = Path(cache_dir) / "industry" / "industry.csv"
            if cache_file.exists():
                size_kb = cache_file.stat().st_size / 1024
                print(f"\n缓存文件: {cache_file}")
                print(f"文件大小: {size_kb:.1f} KB")
        else:
            print("\n✗ 未获取到数据")
            
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    exit_code = download_industry_data()
    sys.exit(exit_code)
