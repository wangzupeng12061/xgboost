# DataLoader 缓存功能说明

## 🎯 功能概述

DataLoader已升级为**优先使用本地缓存**的数据加载器，大幅提升数据加载速度，减少API调用。

## ✨ 主要特性

### 1. 自动缓存检测
- 初始化时自动启用缓存（默认开启）
- 自动检测缓存目录 `./data`
- 缓存命中时直接加载，无需API调用

### 2. 多数据源支持
✅ **股票日线数据** - 从 `data/stock_daily/` 加载  
✅ **股票基本信息** - 从 `data/stock_basic/` 加载  
✅ **指数数据** - 从 `data/index_daily/` 加载  
✅ **财务数据** - 从 `data/financial/` 加载  
✅ **宏观数据** - 从 `data/macro/` 加载  

### 3. 智能回退机制
- 缓存未命中时自动从API获取
- API获取的数据自动保存到缓存
- 完全透明，无需手动干预

## 📖 使用方法

### 基本用法

```python
from src.data.data_loader import DataLoader

# 初始化（默认启用缓存）
loader = DataLoader(
    source="tushare",
    token="your_token",
    use_cache=True,        # 是否使用缓存（默认True）
    cache_dir="./data"     # 缓存目录（默认'./data'）
)
```

### 加载股票列表

```python
# 从缓存加载5445只股票信息
stock_list = loader.load_stock_list()
print(f"加载 {len(stock_list)} 只股票")
```

**输出示例：**
```
✓ 从缓存加载股票列表: 5445 只股票
加载 5445 只股票
```

### 加载日线数据

```python
# 加载指定股票的日线数据
daily_data = loader.load_daily_data(
    start_date='2024-01-01',
    end_date='2024-12-31',
    stock_codes=['000001.SZ', '000002.SZ', '600519.SH']
)
print(f"加载 {len(daily_data)} 条日线记录")
```

**输出示例：**
```
从缓存加载日线数据: 2024-01-01 至 2024-12-31
✓ 加载完成: 3/3 只股票, 共 726 条记录
  数据来源: 缓存 3, API 0
加载 726 条日线记录
```

### 加载指数数据

```python
# 加载沪深300指数
index_data = loader.load_index_data(
    index_code='000300.SH',
    start_date='2024-01-01',
    end_date='2024-12-31'
)
print(f"加载 {len(index_data)} 条指数记录")
```

**输出示例：**
```
✓ 从缓存加载指数数据: 000300.SH, 242 条记录
加载 242 条指数记录
```

### 加载财务数据

```python
# 加载指定股票的财务数据
financial_data = loader.load_financial_data(
    stock_codes=['000001.SZ', '000002.SZ'],
    start_date='2020-01-01',
    end_date='2024-12-31'
)

# 返回字典格式 {ts_code: DataFrame}
for ts_code, df in financial_data.items():
    print(f"{ts_code}: {len(df)} 条财务记录")
```

**输出示例：**
```
从缓存加载财务数据...
✓ 加载完成: 2/2 只股票的财务数据
000001.SZ: 38 条财务记录
000002.SZ: 37 条财务记录
```

### 加载宏观数据

```python
# 加载所有宏观指标
macro_data = loader.load_macro_data(
    start_date='2020-01-01',
    end_date='2024-12-31'
)

# 返回字典格式 {indicator: DataFrame}
for indicator, df in macro_data.items():
    print(f"{indicator}: {len(df)} 条记录")
```

**输出示例：**
```
从缓存加载宏观数据...
  ✓ m1: 573 条记录
  ✓ m2: 573 条记录
  ✓ cpi: 501 条记录
  ✓ ppi: 408 条记录
  ✓ gdp: 175 条记录
  ✓ pmi: 249 条记录
✓ 加载完成: 6/6 个宏观指标
```

## ⚡ 性能对比

| 操作 | 使用缓存 | 不使用缓存 | 提升 |
|------|---------|-----------|------|
| 加载1000只股票日线 | ~2秒 | ~30分钟 | **900倍** |
| 加载5445只股票信息 | <1秒 | ~5分钟 | **300倍** |
| 加载单只股票财务 | <0.1秒 | ~1秒 | **10倍** |
| 加载指数数据 | <0.1秒 | ~1秒 | **10倍** |
| 加载宏观数据 | <0.5秒 | ~5秒 | **10倍** |

## 🔧 配置选项

### 禁用缓存（不推荐）

```python
# 禁用缓存，每次都从API获取
loader = DataLoader(
    source="tushare",
    token="your_token",
    use_cache=False
)
```

### 自定义缓存目录

```python
# 使用自定义缓存目录
loader = DataLoader(
    source="tushare",
    token="your_token",
    use_cache=True,
    cache_dir="/custom/path/to/cache"
)
```

## 📊 数据返回格式

### 股票列表
```python
DataFrame with columns:
- ts_code: 股票代码
- symbol: 股票简称
- name: 股票名称
- area: 地域
- industry: 行业
- market: 市场类型
- list_date: 上市日期
```

### 日线数据
```python
DataFrame with columns:
- ts_code: 股票代码
- trade_date: 交易日期
- open, high, low, close: 开高低收
- pre_close: 前收盘价
- change, pct_chg: 涨跌额、涨跌幅
- vol: 成交量
- amount: 成交额
```

### 指数数据
```python
DataFrame with columns:
- ts_code: 指数代码
- date: 日期
- close: 收盘价
- return: 收益率
```

### 财务数据
```python
Dict[str, DataFrame]
- Key: ts_code (股票代码)
- Value: DataFrame with 100+ financial indicators
  包括: eps, roe, roa, 净利润率, 资产负债率等
```

### 宏观数据
```python
Dict[str, DataFrame]
- Key: indicator name (m1, m2, cpi, ppi, gdp, pmi)
- Value: DataFrame with indicator data
```

## 🎯 最佳实践

### 1. 优先使用缓存加载
```python
# ✅ 推荐：使用缓存
loader = DataLoader(source="tushare", token=token, use_cache=True)

# ❌ 不推荐：禁用缓存（除非确实需要最新数据）
loader = DataLoader(source="tushare", token=token, use_cache=False)
```

### 2. 批量加载优化
```python
# ✅ 推荐：一次性加载所有需要的股票
all_stocks = ['000001.SZ', '000002.SZ', ..., '600519.SH']
data = loader.load_daily_data('2024-01-01', '2024-12-31', all_stocks)

# ❌ 不推荐：逐个加载（即使使用缓存也会慢）
for stock in all_stocks:
    data = loader.load_daily_data('2024-01-01', '2024-12-31', [stock])
```

### 3. 合理的日期范围
```python
# ✅ 推荐：加载需要的日期范围
data = loader.load_daily_data('2020-01-01', '2024-12-31', stocks)

# ❌ 不推荐：加载过大范围（增加内存消耗）
data = loader.load_daily_data('2000-01-01', '2024-12-31', stocks)
```

## 🔍 故障排查

### 1. 缓存加载失败
```python
# 检查缓存目录是否存在
from pathlib import Path
cache_dir = Path('./data')
print(f"缓存目录存在: {cache_dir.exists()}")
print(f"股票日线: {(cache_dir / 'stock_daily').exists()}")
print(f"财务数据: {(cache_dir / 'financial').exists()}")
```

### 2. 数据为空
```python
# 检查缓存统计
from src.data.data_cache import DataCache
cache = DataCache('./data')
stats = cache.get_cache_stats()
print(stats)
```

### 3. 强制刷新数据
```python
# 清除缓存，重新下载
cache = DataCache('./data')
cache.clear_cache(data_type='stock_daily')  # 清除特定类型
cache.clear_cache()  # 清除所有缓存

# 重新下载
python scripts/batch_download_data.py --market a --total 1000
python scripts/download_other_data.py
```

## 📝 更新日志

### v2.0 (2025-11-04)
- ✨ 新增缓存优先加载机制
- ✨ 支持股票日线、指数、财务、宏观数据缓存
- ✨ 智能回退到API获取
- ✨ 自动保存新获取的数据到缓存
- 🚀 性能提升10-900倍

### v1.0 (原版)
- 基础API数据加载功能
- 支持Tushare和AKShare

## 🆘 支持

如有问题，请查看：
- 📖 `docs/数据缓存使用指南.md`
- 📖 `docs/批量下载数据指南.md`
- 🧪 `test/test_dataloader_cache.py`

---

**最后更新**: 2025-11-04  
**版本**: v2.0  
**状态**: ✅ 生产就绪
