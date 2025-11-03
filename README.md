# XGBoost多因子选股系统

基于XGBoost机器学习算法的量化选股系统，实现从数据获取、因子计算、模型训练到回测评估的完整流程。

## 项目结构

```
xgboost/
├── config/                    # 配置文件
│   ├── config.yaml           # 主配置文件
│   └── factor_config.json    # 因子配置
├── docs/                      # 文档
│   ├── XGBoost多因子选股项目文档.md
│   ├── 完整使用指南.md
│   └── ...
├── src/                       # 源代码
│   ├── data/                 # 数据模块
│   │   ├── data_loader.py    # 数据加载
│   │   └── data_processor.py # 数据处理
│   ├── factors/              # 因子模块
│   │   ├── factor_calculator_part1.py
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

# 升级pip
python -m pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置文件

编辑 `config/config.yaml` 设置数据源、回测参数等。

### 3. 运行程序

```bash
python main.py
```

## 主要功能

- **数据获取**: 支持Tushare、AKShare等数据源
- **因子工程**: 60+个量价因子，包括技术、基本面、量化因子
- **智能选股**: XGBoost机器学习模型
- **回测系统**: 完整的回测引擎，支持多种策略
- **绩效分析**: 全面的绩效指标和可视化报告

## 技术栈

- Python 3.8+
- XGBoost 1.7.6
- pandas, numpy, scikit-learn
- matplotlib, seaborn, plotly

## 文档

详细文档请查看 `docs/` 目录：
- [项目文档](docs/XGBoost多因子选股项目文档.md)
- [使用指南](docs/完整使用指南.md)

## 许可证

见 LICENSE 文件
