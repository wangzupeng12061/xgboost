# Scripts 目录

本目录包含用于批量下载和管理股票数据的脚本工具。

## 📋 脚本列表

### 1. batch_download_data.py
**功能**: 批量下载多市场股票数据（A股/港股/美股）

**特点**:
- 支持分批次下载，避免API限流
- 自动断点续传
- 智能重试机制
- 实时进度显示

**使用方法**:
```bash
# 下载A股1000只，2020-2025数据
python scripts/batch_download_data.py \
    --market a \
    --total 1000 \
    --batch-size 50 \
    --start-date 2020-01-01 \
    --end-date 2025-11-04
```

### 2. download_data.sh
**功能**: 交互式批量下载脚本

**特点**:
- 友好的交互式界面
- 自动参数配置
- 一键启动下载

**使用方法**:
```bash
chmod +x scripts/download_data.sh
./scripts/download_data.sh
```

## 🚀 快速开始

### 方法1: 交互式下载（推荐新手）

```bash
./scripts/download_data.sh
```

然后按提示选择：
1. 市场类型（A股/港股/美股）
2. 下载数量
3. 批次大小
4. 日期范围

### 方法2: 命令行下载（推荐高级用户）

```bash
# A股Top1000，2020-2025
python scripts/batch_download_data.py \
    --market a \
    --total 1000 \
    --batch-size 50 \
    --start-date 2020-01-01 \
    --end-date 2025-11-04

# 港股Top500，2020-2025
python scripts/batch_download_data.py \
    --market hk \
    --total 500 \
    --batch-size 50 \
    --start-date 2020-01-01 \
    --end-date 2025-11-04

# 全部市场，1000只
python scripts/batch_download_data.py \
    --market all \
    --total 1000 \
    --batch-size 50 \
    --start-date 2020-01-01 \
    --end-date 2025-11-04
```

## 📊 典型使用场景

### 场景1: 快速测试（3-5分钟）
```bash
python scripts/batch_download_data.py \
    --market a \
    --total 100 \
    --batch-size 50 \
    --start-date 2024-01-01 \
    --end-date 2025-11-04
```

### 场景2: 标准下载（30-40分钟）
```bash
python scripts/batch_download_data.py \
    --market a \
    --total 1000 \
    --batch-size 50 \
    --start-date 2020-01-01 \
    --end-date 2025-11-04
```

### 场景3: 后台运行（推荐大规模下载）
```bash
nohup python scripts/batch_download_data.py \
    --market a \
    --total 2000 \
    --batch-size 50 \
    --start-date 2020-01-01 \
    --end-date 2025-11-04 \
    > download.log 2>&1 &

# 查看进度
tail -f download.log
```

## 📈 参数说明

| 参数 | 说明 | 默认值 | 范围 |
|------|------|--------|------|
| --market | 市场类型 | a | a, hk, us, all |
| --total | 股票数量 | 1000 | 1-10000 |
| --batch-size | 批次大小 | 50 | 10-100 |
| --start-date | 开始日期 | 2020-01-01 | YYYY-MM-DD |
| --end-date | 结束日期 | 2025-11-04 | YYYY-MM-DD |
| --token | Tushare Token | 从config读取 | - |
| --cache-dir | 缓存目录 | ./data | 任意路径 |

## ⏱️ 时间估算

| 股票数 | 年份 | 批次数 | 预计时间 |
|--------|------|--------|----------|
| 100 | 1年 | 2 | 3-5分钟 |
| 500 | 5年 | 10 | 15-20分钟 |
| 1000 | 5年 | 20 | 30-40分钟 |
| 2000 | 5年 | 40 | 60-80分钟 |

## 💾 数据大小

| 场景 | 估算大小 |
|------|---------|
| 100只×5年 | ~25MB |
| 500只×5年 | ~125MB |
| 1000只×5年 | ~250MB |
| 2000只×5年 | ~500MB |

## 🛡️ 限流保护

脚本内置完善的API限流保护：

1. **频率控制**: 每次调用间隔0.5秒
2. **批次管理**: 批次间等待60秒
3. **智能重试**: 失败自动重试3次
4. **限流检测**: 自动识别并等待

## 🔧 故障排除

### 问题1: Token错误
```bash
# 手动指定token
python scripts/batch_download_data.py --token your_token_here
```

### 问题2: 美股下载失败
美股需要Tushare高级权限，建议只下载A股或港股。

### 问题3: 下载中断
重新运行相同命令，已下载数据会自动跳过（断点续传）。

### 问题4: API限流
脚本会自动处理，无需手动干预。

## 📚 相关文档

- [批量下载数据指南](../docs/批量下载数据指南.md) - 详细使用说明
- [数据缓存使用指南](../docs/数据缓存使用指南.md) - 缓存管理
- [data/README.md](../data/README.md) - 数据目录说明

## ✅ 检查清单

使用前确认：
- [ ] 已安装依赖：`pip install -r requirements.txt`
- [ ] 已配置Token：在`config/config.yaml`中
- [ ] 已创建data目录：`mkdir -p data`
- [ ] 网络稳定
- [ ] 磁盘空间充足（建议预留1GB+）

## 🎯 推荐流程

1. **小规模测试** (5分钟)
   ```bash
   python scripts/batch_download_data.py --market a --total 50 --start-date 2024-01-01
   ```

2. **检查结果**
   ```bash
   ls -lh data/stock_daily/ | wc -l
   du -sh data/
   ```

3. **正式下载** (30-40分钟)
   ```bash
   nohup python scripts/batch_download_data.py \
       --market a --total 1000 \
       --start-date 2020-01-01 \
       > download.log 2>&1 &
   ```

4. **监控进度**
   ```bash
   tail -f download.log
   tail -f logs/*.log
   ```

## 🚀 高级技巧

### 1. 并行下载不同市场

```bash
# 终端1: 下载A股
python scripts/batch_download_data.py --market a --total 1000 &

# 终端2: 下载港股
python scripts/batch_download_data.py --market hk --total 500 &
```

### 2. 分时段下载

```bash
# 2020-2022
python scripts/batch_download_data.py --start-date 2020-01-01 --end-date 2022-12-31

# 2023-2025
python scripts/batch_download_data.py --start-date 2023-01-01 --end-date 2025-11-04
```

### 3. 增量更新

```bash
# 只更新最近3个月
python scripts/batch_download_data.py \
    --start-date 2025-08-01 \
    --end-date 2025-11-04
```

---

**提示**: 首次使用建议先进行小规模测试！
