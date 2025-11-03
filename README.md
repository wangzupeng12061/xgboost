# XGBoost多因子选股系统

基于XGBoost机器学习算法的量化选股系统，实现从数据获取、因子计算、模型训练到回测评估的完整流程。

## 🌟 项目特点

- ✅ **完整流程**: 数据获取 → 因子计算 → 模型训练 → 回测评估 → 可视化分析
- ✅ **88个量化因子**: 覆盖8大类因子（技术、动量、波动率、价值、成长、质量、流动性、反转）
- ✅ **智能因子筛选**: 基于IC/ICIR自动筛选有效因子
- ✅ **多分类支持**: 支持二分类、多分类、回归等多种标签构建方法
- ✅ **滚动训练回测**: 模拟真实交易环境的时序验证
- ✅ **全面绩效评估**: 22项专业指标（收益、风险、风险调整收益、相对基准等）
- ✅ **专业可视化**: 7种图表全面展示策略表现
- ✅ **灵活配置**: YAML配置文件，一键切换参数

## 📊 实现功能

### 核心模块

| 模块 | 功能 | 状态 |
|------|------|------|
| **数据加载** | Tushare接口，支持股票、指数、财务、宏观数据 | ✅ 完成 |
| **因子计算** | 88个量化因子，8大类别 | ✅ 完成 |
| **因子处理** | 去极值、标准化、中性化、缺失值处理 | ✅ 完成 |
| **因子筛选** | IC/ICIR筛选，相关性去重 | ✅ 完成 |
| **标签构建** | 7种标签方法，支持多分类/回归 | ✅ 完成 |
| **模型训练** | XGBoost，支持多分类 | ✅ 完成 |
| **超参数优化** | 网格搜索、随机搜索、贝叶斯优化 | ✅ 完成 |
| **回测引擎** | 滚动训练，模拟真实交易 | ✅ 完成 |
| **绩效评估** | 22项专业指标 | ✅ 完成 |
| **可视化** | 7种专业图表 | ✅ 完成 |

### 因子体系

**88个因子，8大类别：**

1. **技术指标类** (14个): MA, EMA, MACD, KDJ, RSI, BOLL等
2. **动量类** (12个): ROC, MTM, 新高新低比例等
3. **波动率类** (11个): ATR, 历史波动率、收益波动等
4. **价值类** (11个): PE, PB, PS, PCF, EV/EBITDA等
5. **成长类** (10个): 营收增长、利润增长、ROE增长等
6. **质量类** (10个): ROE, ROA, 毛利率、周转率等
7. **流动性类** (10个): 换手率、成交量、流通市值等
8. **反转类** (10个): 各周期收益反转因子

### 标签构建方法

支持7种标签构建方法：

1. **二分类标签**: 基于阈值划分涨跌
2. **多分类标签**: 分位数分层（弱/中/强）
3. **回归标签**: 直接预测收益率
4. **排名标签**: 收益率排名
5. **超额收益标签**: 相对基准的超额收益
6. **多周期标签**: 不同时间窗口
7. **波动率调整标签**: 考虑风险调整

### 绩效评估指标

**22项专业指标：**

- **收益指标** (3项): 累计收益率、年化收益率、日均收益率
- **风险指标** (6项): 波动率、下行波动率、最大回撤、回撤天数、VaR、CVaR
- **风险调整收益** (4项): 夏普比率、索提诺比率、卡玛比率、Omega比率
- **相对基准** (7项): Alpha、Beta、信息比率、跟踪误差、主动收益、上下行捕获率
- **交易统计** (2项): 胜率、盈亏比

### 可视化图表

**7种专业图表：**

1. 净值曲线图 - 策略vs基准对比
2. 回撤曲线图 - 风险分析
3. 收益率分布图 - 直方图+箱线图
4. 滚动指标图 - 收益/波动/夏普
5. 月度收益热力图 - 季节性分析
6. 因子重要性图 - Top 20因子
7. 综合看板 - 一页汇总

## 项目结构

```
xgboost/
├── config/                    # 配置文件
│   ├── config.yaml           # 主配置文件
│   └── factor_config.json    # 因子配置
├── docs/                      # 文档目录
│   ├── XGBoost多因子选股项目文档.md
│   ├── 完整使用指南.md
│   ├── 绩效评估模块说明.md
│   ├── 可视化模块说明.md
│   └── ...
├── src/                       # 源代码
│   ├── data/                 # 数据模块
│   │   ├── data_loader.py    # 数据加载（Tushare）
│   │   └── data_processor.py # 数据处理
│   ├── factors/              # 因子模块
│   │   ├── factor_calculator.py  # 因子计算（88个因子）
│   │   ├── factor_processor.py   # 因子预处理
│   │   └── factor_selector.py    # 因子筛选
│   ├── model/                # 模型模块
│   │   ├── label_builder.py  # 标签构建（7种方法）
│   │   ├── model_tuner.py    # 超参数优化
│   │   └── xgb_model.py      # XGBoost模型
│   ├── backtest/             # 回测模块
│   │   ├── backtester.py     # 回测引擎
│   │   ├── evaluator.py      # 绩效评估（22项指标）
│   │   ├── portfolio_manager.py  # 组合管理
│   │   └── stock_selector.py     # 选股器
│   └── utils/                # 工具模块
│       ├── logger.py         # 日志工具
│       └── visualization.py  # 可视化（7种图表）
├── results/                  # 结果目录
│   ├── figures/             # 图表
│   ├── models/              # 模型文件
│   └── reports/             # 报告
├── logs/                     # 日志目录
├── main.py                   # 主程序入口
├── requirements.txt          # 依赖清单
└── README.md                # 本文件
```

## 🚀 快速开始
│   │   ├── factor_calculator_part2.py
│   │   ├── factor_processor.py
│   │   └── factor_selector.py
│   ├── model/                # 模型模块
│   │   ├── label_builder.py  # 标签构建
│   │   ├── model_tuner.py    # 超参数调优
│   │   └── xgb_model.py      # XGBoost模型
│   ├── backtest/             # 回测模块
│   │   ├── backtester.py     # 回测引擎
│   │   ├── evaluator.py      # 绩效评估
│   │   ├── portfolio_manager.py  # 组合管理
│   │   └── stock_selector.py     # 选股器
│   └── utils/                # 工具模块
│       ├── logger.py         # 日志工具
│       └── visualization.py  # 可视化
├── main.py                   # 主程序入口
├── requirements.txt          # 依赖清单
└── README.md                # 本文件
```

## 快速开始

### 1. 环境设置

```bash
# 创建虚拟环境
python3 -m venv .venv

# 激活虚拟环境
source .venv/bin/activate  # macOS/Linux
# 或
.venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

**主要依赖：**
- Python 3.9+
- pandas 1.5.3
- numpy 1.23.5
- xgboost 2.0.0
- scikit-learn 1.3.0
- tushare 1.2.89
- matplotlib 3.7.1
- seaborn 0.12.2

### 2. 配置Tushare Token

1. 注册Tushare账号：https://tushare.pro/register
2. 获取token
3. 编辑 `config/config.yaml`，填入token：

```yaml
data:
  source: "tushare"
  token: "your_tushare_token_here"  # 填入你的token
  start_date: "2024-01-01"
  end_date: "2024-10-31"
```

### 3. 运行程序

```bash
python main.py
```

**完整流程：**
1. 数据加载（股票、宏观、财务数据）
2. 因子计算（88个因子）
3. 因子预处理（去极值、标准化、中性化）
4. 因子筛选（IC/ICIR筛选）
5. 标签构建（多分类标签）
6. 模型训练（XGBoost）
7. 回测（滚动训练）
8. 绩效评估（22项指标）
9. 可视化（7种图表）
10. 结果保存

### 4. 查看结果

运行完成后，结果保存在：

- **图表**: `results/figures/`
  - equity_curve_*.png - 净值曲线
  - drawdown_*.png - 回撤分析
  - returns_dist_*.png - 收益分布
  - rolling_metrics_*.png - 滚动指标
  - monthly_returns_*.png - 月度收益
  - feature_importance_*.png - 因子重要性
  - dashboard_*.png - 综合看板

- **报告**: `results/reports/`
  - performance_metrics_*.csv - 绩效指标数据
  - performance_report_*.txt - 文本报告

- **模型**: `results/models/`
  - model_*.pkl - 训练好的模型
  - model_*_metadata.json - 模型元数据

- **日志**: `logs/`
  - xgboost_stock_*.log - 运行日志

## 📖 详细文档

- [完整使用指南](docs/完整使用指南.md) - 详细的使用说明
- [XGBoost多因子选股项目文档](docs/XGBoost多因子选股项目文档.md) - 技术文档
- [绩效评估模块说明](docs/绩效评估模块说明.md) - 22项指标详解
- [可视化模块说明](docs/可视化模块说明.md) - 7种图表说明

## ⚙️ 配置说明

主配置文件：`config/config.yaml`

### 数据配置

```yaml
data:
  source: "tushare"           # 数据源（tushare/akshare）
  token: "your_token_here"    # Tushare token
  start_date: "2024-01-01"    # 数据开始日期
  end_date: "2024-10-31"      # 数据结束日期
  save_path: "./data"         # 数据保存路径
```

**推荐设置：**
- **小规模测试**：30只股票，1年数据（2024）
- **中规模测试**：50只股票，2年数据（2023-2024）
- **大规模测试**：需要Tushare付费版本（避免API限流）

### 回测配置

```yaml
backtest:
  start_date: "2024-02-01"    # 回测开始日期
  end_date: "2024-10-31"      # 回测结束日期
  initial_capital: 1000000    # 初始资金（100万）
  benchmark: "000300.SH"      # 基准指数（沪深300）
  top_n: 10                   # 选股数量
  rebalance_freq: "M"         # 调仓频率（M=月度，W=周度）
```

### 因子筛选配置

```yaml
factor_selection:
  method: "information_coefficient"  # 筛选方法
  ic_threshold: 0.02                 # IC阈值
  icir_threshold: 0.5                # ICIR阈值
  top_n: 30                          # 保留因子数量
```

### 模型配置

```yaml
model:
  n_estimators: 100          # 决策树数量
  max_depth: 5               # 最大深度
  learning_rate: 0.1         # 学习率
  min_child_weight: 1        # 最小子节点权重
  subsample: 0.8             # 样本采样比例
  colsample_bytree: 0.8      # 特征采样比例
```

## 🎯 最佳实践

### 1. API限流管理

Tushare免费版限制：**200次调用/分钟**

**建议：**
- 单次运行不超过30只股票
- 数据时间范围不超过1年
- 或升级到Tushare付费版本

### 2. 因子选择

默认计算88个因子，建议：
- 初次运行使用所有因子
- 根据因子重要性图，保留有效因子
- 编辑 `config/factor_config.json` 禁用低效因子

### 3. 回测周期

- **短期测试**（1-2年）：验证策略逻辑
- **中期测试**（3-5年）：观察稳定性
- **长期测试**（5-10年）：验证适应性

### 4. 性能优化

- 增大 `train_period`（训练窗口）提高模型稳定性
- 减少 `top_n`（选股数量）降低换手率
- 调整 `rebalance_freq` 平衡收益和成本

## 🔧 故障排除

### 问题1：API限流错误

```
TushareNetworkException: 接口限流，请稍后重试
```

**解决方案：**
- 检查是否超过30只股票
- 缩短数据时间范围
- 等待1分钟后重试

### 问题2：程序卡住不动

**可能原因：**
- API请求过多导致限流
- 数据加载时间过长

**解决方案：**
1. 检查日志：`tail -f logs/xgboost_stock_*.log`
2. 减少股票数量到30只
3. 重启程序

### 问题3：因子计算失败

```
KeyError: 'close'
```

**解决方案：**
- 检查数据是否完整加载
- 确认股票代码有效
- 延长数据开始日期（留出因子计算窗口）

### 问题4：图表不显示

**解决方案：**
- 图表自动保存到 `results/figures/`
- 使用图片查看器打开PNG文件
- 不需要交互式显示

## 📊 示例结果

基于30只股票、2024年数据的回测结果：

**绩效指标：**
- 累计收益率：-8.14%
- 年化收益率：-11.86%
- 最大回撤：-16.42%
- 夏普比率：-0.86
- 胜率：42.11%

**与沪深300对比：**
- 超额收益：根据市场走势动态变化
- 信息比率：衡量超额收益的稳定性
- 跟踪误差：策略与基准的偏离度

**图表输出：**
7种专业图表自动生成，每张图300 DPI高清PNG格式

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📝 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 📮 联系方式

如有问题或建议，请提交Issue或联系项目维护者。

## 🙏 致谢

- [Tushare](https://tushare.pro/) - 提供金融数据接口
- [XGBoost](https://xgboost.readthedocs.io/) - 机器学习框架
- 所有为本项目做出贡献的开发者

## 文档

详细文档请查看 `docs/` 目录：
- [项目文档](docs/XGBoost多因子选股项目文档.md)
- [使用指南](docs/完整使用指南.md)

## 许可证

见 LICENSE 文件
