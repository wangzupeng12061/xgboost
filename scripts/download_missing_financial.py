#!/usr/bin/env python3
"""
下载缺失的财务数据
"""

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
from pathlib import Path
import time
import yaml
from src.data.data_cache import DataCache
from src.data.data_loader import DataLoader


def download_missing_financial():
    """下载缺失的财务数据"""
    
    print("=" * 60)
    print("下载缺失的财务数据")
    print("=" * 60)
    
    # 加载配置
    config_path = Path("config/config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 初始化缓存和数据加载器
    cache = DataCache(cache_dir="./data", expire_days=0)
    loader = DataLoader(
        source="tushare",
        token=config['data']['token'],
        use_cache=False
    )
    
    # 找出缺失的股票
    stock_basic = pd.read_csv('data/stock_basic/stock_basic.csv')
    all_stocks = set(stock_basic['ts_code'].tolist())
    
    financial_files = set([f.stem for f in Path('data/financial').glob('*.csv')])
    missing_stocks = sorted(list(all_stocks - financial_files))
    
    print(f"\n找到 {len(missing_stocks)} 只股票缺少财务数据:")
    for code in missing_stocks:
        print(f"  - {code}")
    
    if not missing_stocks:
        print("\n✓ 所有股票都已有财务数据!")
        return 0
    
    # 下载缺失的财务数据
    print(f"\n开始下载...")
    success_count = 0
    fail_count = 0
    
    for i, stock_code in enumerate(missing_stocks, 1):
        print(f"\n[{i}/{len(missing_stocks)}] 下载 {stock_code}...", end=" ", flush=True)
        
        try:
            # 获取财务数据
            df = loader.pro.fina_indicator(
                ts_code=stock_code,
                fields="ts_code,end_date,roe,roa,gross_profit_margin,net_profit_margin,"
                       "debt_asset_ratio,current_ratio,quick_ratio,equity_ratio,"
                       "total_revenue,net_profit,total_assets,total_liab"
            )
            
            if df is not None and len(df) > 0:
                # 保存到缓存
                cache.save_financial_data(stock_code, df)
                print(f"✓ ({len(df)} 条记录)")
                success_count += 1
            else:
                print("✗ (无数据)")
                fail_count += 1
            
            # API限速
            time.sleep(0.31)
            
        except Exception as e:
            print(f"✗ 失败: {e}")
            fail_count += 1
            time.sleep(0.31)
    
    # 显示结果
    print("\n" + "=" * 60)
    print("下载完成!")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print("=" * 60)
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    exit_code = download_missing_financial()
    sys.exit(exit_code)
