"""
数据缓存使用示例
演示如何使用DataCache类管理股票数据缓存
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_cache import DataCache
from src.utils.logger import setup_logger
import pandas as pd

# 设置日志
logger = setup_logger("data_cache_example")


def example_basic_usage():
    """基本使用示例"""
    print("\n" + "=" * 60)
    print("示例1: 基本使用")
    print("=" * 60)

    # 创建缓存管理器
    cache = DataCache(cache_dir="./data", expire_days=7)

    # 模拟数据
    sample_data = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=10),
            "open": [10.0 + i for i in range(10)],
            "high": [11.0 + i for i in range(10)],
            "low": [9.0 + i for i in range(10)],
            "close": [10.5 + i for i in range(10)],
            "volume": [1000000 + i * 10000 for i in range(10)],
        }
    )

    # 保存缓存
    print("\n保存股票日线数据缓存...")
    cache.save_stock_daily("000001.SZ", sample_data)

    # 读取缓存
    print("\n读取股票日线数据缓存...")
    cached_data = cache.get_stock_daily("000001.SZ")
    if cached_data is not None:
        print(f"✓ 读取成功，数据量: {len(cached_data)}")
        print(cached_data.head())
    else:
        print("✗ 读取失败")


def example_cache_stats():
    """缓存统计示例"""
    print("\n" + "=" * 60)
    print("示例2: 缓存统计信息")
    print("=" * 60)

    cache = DataCache(cache_dir="./data", expire_days=7)

    # 获取缓存统计
    stats = cache.get_cache_stats()

    print("\n缓存统计信息:")
    print(f"  缓存目录: {stats['cache_dir']}")
    print(f"  过期天数: {stats['expire_days']} 天")
    print(f"  股票日线数据文件: {stats['stock_daily_count']}")
    print(f"  股票基本信息: {'已缓存' if stats['stock_basic_exists'] else '未缓存'}")
    print(f"  指数日线数据文件: {stats['index_daily_count']}")
    print(f"  宏观数据文件: {stats['macro_count']}")
    print(f"  财务数据文件: {stats['financial_count']}")
    print(f"  总大小: {stats['total_size_mb']} MB")


def example_with_dataloader():
    """配合DataLoader使用示例"""
    print("\n" + "=" * 60)
    print("示例3: 配合DataLoader使用")
    print("=" * 60)

    try:
        from src.data.data_loader import DataLoader

        # 创建DataLoader（带缓存）
        loader = DataLoader(source="tushare", token="your_token_here")

        # 如果DataLoader支持缓存，可以这样使用
        print("\n提示: 在DataLoader中集成缓存后，数据将自动缓存")
        print("      首次加载从API获取，后续加载从本地缓存读取")

    except ImportError:
        print("✗ 需要先实现DataLoader与缓存的集成")


def example_macro_data():
    """宏观数据缓存示例"""
    print("\n" + "=" * 60)
    print("示例4: 宏观数据缓存")
    print("=" * 60)

    cache = DataCache(cache_dir="./data", expire_days=0)  # 永不过期

    # 模拟宏观数据
    macro_data = pd.DataFrame(
        {
            "month": ["202401", "202402", "202403"],
            "m1_yoy": [5.2, 5.5, 5.8],
            "m2_yoy": [8.7, 8.9, 9.1],
        }
    )

    # 保存宏观数据
    print("\n保存宏观数据缓存...")
    cache.save_macro_data("m1", macro_data[["month", "m1_yoy"]])
    cache.save_macro_data("m2", macro_data[["month", "m2_yoy"]])

    # 读取宏观数据
    print("\n读取宏观数据缓存...")
    m1_data = cache.get_macro_data("m1")
    m2_data = cache.get_macro_data("m2")

    if m1_data is not None:
        print(f"✓ M1数据读取成功，数据量: {len(m1_data)}")
        print(m1_data)

    if m2_data is not None:
        print(f"✓ M2数据读取成功，数据量: {len(m2_data)}")
        print(m2_data)


def example_cache_management():
    """缓存管理示例"""
    print("\n" + "=" * 60)
    print("示例5: 缓存管理")
    print("=" * 60)

    cache = DataCache(cache_dir="./data", expire_days=7)

    print("\n1. 查看缓存统计")
    stats = cache.get_cache_stats()
    print(f"   当前缓存大小: {stats['total_size_mb']} MB")

    print("\n2. 清除特定类型缓存")
    print("   示例命令: cache.clear_cache('stock_daily')")

    print("\n3. 清除所有缓存")
    print("   示例命令: cache.clear_cache()")

    print("\n4. 设置缓存过期时间")
    print("   expire_days=0: 永不过期")
    print("   expire_days=7: 7天后过期")
    print("   expire_days=30: 30天后过期")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("数据缓存使用示例")
    print("=" * 60)

    # 运行所有示例
    example_basic_usage()
    example_cache_stats()
    example_macro_data()
    example_cache_management()
    example_with_dataloader()

    print("\n" + "=" * 60)
    print("所有示例运行完成!")
    print("=" * 60 + "\n")
