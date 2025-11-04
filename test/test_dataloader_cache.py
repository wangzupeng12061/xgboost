"""
测试DataLoader缓存功能
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
from src.data.data_loader import DataLoader


def test_dataloader_cache():
    """测试DataLoader使用缓存"""
    
    # 读取配置
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        token = config["data"]["token"]
    
    print("=" * 70)
    print("测试DataLoader缓存功能")
    print("=" * 70)
    
    # 初始化DataLoader（启用缓存）
    loader = DataLoader(source="tushare", token=token, use_cache=True, cache_dir="./data")
    
    # 1. 测试股票列表
    print("\n1️⃣  测试股票列表...")
    stock_list = loader.load_stock_list()
    print(f"股票列表数量: {len(stock_list)}")
    print(f"前5只股票:\n{stock_list.head()}")
    
    # 2. 测试日线数据
    print("\n2️⃣  测试日线数据...")
    # 加载前10只股票的日线数据
    test_stocks = ['000001.SZ', '000002.SZ', '600000.SH', '600519.SH', '000858.SZ']
    daily_data = loader.load_daily_data(
        start_date='2024-01-01',
        end_date='2024-12-31',
        stock_codes=test_stocks
    )
    print(f"日线数据记录数: {len(daily_data)}")
    if len(daily_data) > 0:
        print(f"数据样例:\n{daily_data.head()}")
    
    # 3. 测试指数数据
    print("\n3️⃣  测试指数数据...")
    index_data = loader.load_index_data(
        index_code='000300.SH',
        start_date='2024-01-01',
        end_date='2024-12-31'
    )
    print(f"指数数据记录数: {len(index_data)}")
    if len(index_data) > 0:
        print(f"数据样例:\n{index_data.head()}")
    
    # 4. 测试财务数据
    print("\n4️⃣  测试财务数据...")
    financial_data = loader.load_financial_data(stock_codes=test_stocks[:3])
    print(f"财务数据股票数: {len(financial_data)}")
    for ts_code, df in list(financial_data.items())[:2]:
        print(f"  {ts_code}: {len(df)} 条记录")
    
    # 5. 测试宏观数据
    print("\n5️⃣  测试宏观数据...")
    macro_data = loader.load_macro_data()
    print(f"宏观指标数: {len(macro_data)}")
    for indicator, df in macro_data.items():
        print(f"  {indicator}: {len(df)} 条记录")
    
    print("\n" + "=" * 70)
    print("✅ 所有测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    test_dataloader_cache()
