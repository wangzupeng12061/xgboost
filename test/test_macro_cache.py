"""
宏观数据缓存功能单元测试

测试DataLoader的宏观数据加载功能，确保：
1. 缓存路径返回正确的DataFrame格式
2. 日期列规范化正常工作
3. 多个指标正确合并
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from pathlib import Path
from src.data.data_loader import DataLoader
from src.data.data_cache import DataCache


def test_macro_cache_returns_dataframe():
    """测试宏观数据缓存加载返回DataFrame而非dict"""
    
    # 检查缓存目录是否存在
    cache_dir = Path('./data/macro')
    if not cache_dir.exists():
        print("  ⚠️  宏观数据缓存目录不存在，跳过测试")
        return
    
    # 检查是否有缓存文件
    cache_files = list(cache_dir.glob('*.csv'))
    if len(cache_files) == 0:
        print("  ⚠️  宏观数据缓存为空，跳过测试")
        return
    
    # 初始化DataLoader（使用缓存）
    loader = DataLoader(
        source='tushare',
        token='dummy_token',  # 测试中不会真正调用API
        use_cache=True,
        cache_dir='./data'
    )
    
    # 加载宏观数据
    macro_data = loader.load_macro_data(
        start_date='2020-01-01',
        end_date='2025-11-04'
    )
    
    # 断言：返回值应该是DataFrame
    assert isinstance(macro_data, pd.DataFrame), \
        f"Expected DataFrame, got {type(macro_data)}"
    
    # 断言：DataFrame应该包含'date'列
    assert 'date' in macro_data.columns, \
        f"Expected 'date' column in macro_data, got columns: {macro_data.columns.tolist()}"
    
    # 断言：date列应该是datetime类型
    assert pd.api.types.is_datetime64_any_dtype(macro_data['date']), \
        f"Expected 'date' column to be datetime, got {macro_data['date'].dtype}"
    
    # 断言：应该有多个指标列（除了date）
    indicator_cols = [col for col in macro_data.columns if col != 'date']
    assert len(indicator_cols) > 0, \
        "Expected at least one indicator column besides 'date'"
    
    print(f"✓ 测试通过：宏观数据返回DataFrame，包含 {len(indicator_cols)} 个指标")
    print(f"  指标列: {indicator_cols}")
    print(f"  数据形状: {macro_data.shape}")
    print(f"  日期范围: {macro_data['date'].min()} 至 {macro_data['date'].max()}")


def test_macro_cache_date_normalization():
    """测试日期列规范化功能"""
    
    cache_dir = Path('./data/macro')
    if not cache_dir.exists():
        print("  ⚠️  宏观数据缓存目录不存在，跳过测试")
        return
    
    # 初始化DataCache直接测试
    cache = DataCache(cache_dir='./data', expire_days=0)
    
    # 测试读取单个指标
    indicators = ['m1', 'm2', 'cpi', 'ppi', 'gdp', 'pmi']
    loaded_count = 0
    
    for indicator in indicators:
        df = cache.get_macro_data(indicator)
        if df is not None and len(df) > 0:
            loaded_count += 1
            
            # 检查是否有日期相关列
            date_cols = [col for col in df.columns 
                        if col.lower() in ['date', 'month', 'quarter', 'year']]
            
            assert len(date_cols) > 0, \
                f"Indicator {indicator} should have at least one date column"
            
            print(f"  ✓ {indicator}: 包含日期列 {date_cols}, 形状 {df.shape}")
    
    print(f"✓ 测试通过：成功加载 {loaded_count}/{len(indicators)} 个宏观指标缓存")


def test_macro_cache_merge_logic():
    """测试多指标合并逻辑"""
    
    cache_dir = Path('./data/macro')
    if not cache_dir.exists():
        print("  ⚠️  宏观数据缓存目录不存在，跳过测试")
        return
    
    loader = DataLoader(
        source='tushare',
        token='dummy_token',
        use_cache=True,
        cache_dir='./data'
    )
    
    # 加载宏观数据
    macro_data = loader.load_macro_data(
        start_date='2020-01-01',
        end_date='2025-11-04'
    )
    
    if len(macro_data) == 0:
        print("  ⚠️  宏观数据为空，跳过测试")
        return
    
    # 检查是否按日期排序
    assert macro_data['date'].is_monotonic_increasing, \
        "Macro data should be sorted by date"
    
    # 检查是否有重复的日期（应该没有，因为按date合并）
    duplicate_dates = macro_data[macro_data['date'].duplicated()]
    assert len(duplicate_dates) == 0, \
        f"Found {len(duplicate_dates)} duplicate dates in macro data"
    
    # 检查列名是否唯一
    duplicate_cols = [col for col in macro_data.columns 
                     if macro_data.columns.tolist().count(col) > 1]
    assert len(duplicate_cols) == 0, \
        f"Found duplicate column names: {duplicate_cols}"
    
    print(f"✓ 测试通过：宏观数据合并正确")
    print(f"  唯一日期数: {macro_data['date'].nunique()}")
    print(f"  总行数: {len(macro_data)}")
    print(f"  列数: {len(macro_data.columns)}")


def test_macro_cache_date_filtering():
    """测试日期范围筛选功能"""
    
    cache_dir = Path('./data/macro')
    if not cache_dir.exists():
        print("  ⚠️  宏观数据缓存目录不存在，跳过测试")
        return
    
    loader = DataLoader(
        source='tushare',
        token='dummy_token',
        use_cache=True,
        cache_dir='./data'
    )
    
    # 加载特定日期范围的数据
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    macro_data = loader.load_macro_data(
        start_date=start_date,
        end_date=end_date
    )
    
    if len(macro_data) == 0:
        print("  ⚠️  宏观数据为空，跳过测试")
        return
    
    # 验证日期范围
    min_date = macro_data['date'].min()
    max_date = macro_data['date'].max()
    
    assert min_date >= pd.to_datetime(start_date), \
        f"Min date {min_date} should be >= {start_date}"
    
    assert max_date <= pd.to_datetime(end_date), \
        f"Max date {max_date} should be <= {end_date}"
    
    print(f"✓ 测试通过：日期筛选正确")
    print(f"  请求范围: {start_date} 至 {end_date}")
    print(f"  实际范围: {min_date} 至 {max_date}")


if __name__ == '__main__':
    """直接运行测试"""
    print("\n" + "="*70)
    print("宏观数据缓存功能测试")
    print("="*70 + "\n")
    
    try:
        print("测试1: 返回类型验证")
        print("-" * 70)
        test_macro_cache_returns_dataframe()
        print()
        
        print("测试2: 日期列规范化")
        print("-" * 70)
        test_macro_cache_date_normalization()
        print()
        
        print("测试3: 多指标合并逻辑")
        print("-" * 70)
        test_macro_cache_merge_logic()
        print()
        
        print("测试4: 日期范围筛选")
        print("-" * 70)
        test_macro_cache_date_filtering()
        print()
        
        print("="*70)
        print("✅ 所有测试通过！")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
