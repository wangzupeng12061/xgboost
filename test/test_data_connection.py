"""
测试 Tushare 数据获取功能
"""

import yaml
import tushare as ts
import pandas as pd
from datetime import datetime

def test_tushare_connection():
    """测试 Tushare 连接和数据获取"""
    
    # 加载配置
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    token = config['data']['token']
    
    print("="*60)
    print("测试 Tushare 数据获取")
    print("="*60)
    
    # 设置 token
    ts.set_token(token)
    pro = ts.pro_api()
    
    print(f"\n✓ Token 配置成功")
    print(f"Token: {token[:20]}...")
    
    # 测试1: 获取股票列表
    print("\n" + "-"*60)
    print("测试1: 获取交易日历")
    print("-"*60)
    
    try:
        # 使用交易日历接口（免费用户可用）
        trade_cal = pro.trade_cal(
            exchange='SSE',
            start_date='20241001',
            end_date='20241031',
            is_open='1'
        )
        print(f"✓ 成功获取交易日历")
        print(f"  交易日数量: {len(trade_cal)}")
        print(f"\n交易日列表:")
        print(trade_cal)
    except Exception as e:
        print(f"✗ 获取交易日历失败: {e}")
        return False
    
    # 测试2: 获取单只股票的日线数据
    print("\n" + "-"*60)
    print("测试2: 获取日线数据 (平安银行 000001.SZ)")
    print("-"*60)
    
    try:
        df = pro.daily(
            ts_code='000001.SZ',
            start_date='20240101',
            end_date='20241031'
        )
        print(f"✓ 成功获取日线数据")
        print(f"  数据条数: {len(df)}")
        print(f"\n最近5天数据:")
        print(df.head())
    except Exception as e:
        print(f"✗ 获取日线数据失败: {e}")
        return False
    
    # 测试3: 获取指数数据
    print("\n" + "-"*60)
    print("测试3: 获取指数数据 (沪深300)")
    print("-"*60)
    
    try:
        df_index = pro.index_daily(
            ts_code='000300.SH',
            start_date='20240101',
            end_date='20241031'
        )
        print(f"✓ 成功获取指数数据")
        print(f"  数据条数: {len(df_index)}")
        print(f"\n最近5天数据:")
        print(df_index.head())
    except Exception as e:
        print(f"✗ 获取指数数据失败: {e}")
        return False
    
    # 测试4: 尝试获取更多股票数据
    print("\n" + "-"*60)
    print("测试4: 获取多只股票数据")
    print("-"*60)
    
    try:
        # 获取几只常见股票的数据
        stocks = ['000001.SZ', '600000.SH', '000002.SZ']
        for stock in stocks[:2]:  # 只测试前2只
            df_stock = pro.daily(
                ts_code=stock,
                start_date='20241001',
                end_date='20241031'
            )
            print(f"  {stock}: {len(df_stock)} 条数据")
        print(f"✓ 成功获取多只股票数据")
    except Exception as e:
        print(f"✗ 获取股票数据失败: {e}")
        return False
    
    print("\n" + "="*60)
    print("✅ 所有测试通过！Tushare 数据获取功能正常")
    print("="*60)
    
    return True


if __name__ == "__main__":
    success = test_tushare_connection()
    
    if not success:
        print("\n❌ 测试失败，请检查 token 或网络连接")
    else:
        print("\n🎉 数据源配置成功，可以开始使用项目！")
