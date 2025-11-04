# 因子数据缓存目录

此目录用于存储计算好的因子数据，以便快速回测。

## 文件说明

- `factors_latest.parquet` - 最新的因子数据（自动覆盖）
- `factors_YYYYMMDD_HHMMSS.parquet` - 带时间戳的历史版本

## 使用方法

### 生成缓存
```bash
python main.py  # 运行完整流程，自动保存因子数据
```

### 使用缓存
```bash
python run_backtest_with_model.py  # 自动加载 factors_latest.parquet
```

## 注意事项

1. 因子数据与训练时的配置相关，配置变化后需要重新生成
2. 定期清理旧的时间戳文件以节省磁盘空间
3. `factors_latest.parquet` 始终保持最新，建议保留

## 清理缓存

```bash
# 删除旧版本（保留最新的）
rm cache/factors_202*.parquet

# 完全清空（需要重新运行 main.py）
rm cache/*.parquet
```
