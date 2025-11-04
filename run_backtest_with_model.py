"""
使用已保存的模型进行回测
直接加载训练好的模型，跳过训练步骤，只运行回测和评估
"""

import yaml
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import os
import sys
warnings.filterwarnings('ignore')

# 导入项目模块
from src.data.data_loader import DataLoader
from src.model.xgb_model import XGBoostModel
from src.backtest.stock_selector import StockSelector
from src.backtest.portfolio_manager import PortfolioManager
from src.backtest.backtester import Backtester
from src.backtest.evaluator import PerformanceEvaluator
from src.utils.logger import setup_logger, LoggerContext
from src.utils.visualization_fixed import (
    plot_equity_curve, 
    plot_drawdown, 
    plot_returns_distribution,
    plot_rolling_metrics,
    plot_monthly_returns,
    create_dashboard
)


def load_config(config_path='config/config.yaml'):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main(model_path: str = None):
    """
    主函数 - 使用已保存模型进行回测
    
    Args:
        model_path: 模型文件路径，如果为None则使用最新的模型
    """
    
    print("\n" + "="*80)
    print("XGBoost多因子选股 - 使用已保存模型回测")
    print("="*80)
    
    # 加载配置
    config = load_config()
    
    # 设置日志
    logger = setup_logger(
        name='xgboost_backtest',
        log_dir=config['output']['log_path'],
        level=20
    )
    
    logger.info("="*60)
    logger.info("回测启动（使用已保存模型）")
    logger.info("="*60)
    
    try:
        # ========== 1. 查找或指定模型 ==========
        with LoggerContext(logger, "加载模型"):
            if model_path is None:
                # 查找最新的模型文件
                model_dir = config['output']['model_path']
                model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
                
                if not model_files:
                    logger.error(f"在 {model_dir} 中未找到模型文件！")
                    logger.info("请先运行 main.py 训练模型")
                    return False
                
                # 按时间戳排序，获取最新的
                model_files.sort(reverse=True)
                model_path = os.path.join(model_dir, model_files[0])
                logger.info(f"自动选择最新模型: {model_files[0]}")
            else:
                if not os.path.exists(model_path):
                    logger.error(f"模型文件不存在: {model_path}")
                    return False
                logger.info(f"使用指定模型: {model_path}")
            
            # 加载模型
            model = XGBoostModel()
            model.load_model(model_path)
            logger.info(f"✓ 模型加载成功")
            logger.info(f"  任务类型: {model.task_type}")
            logger.info(f"  特征数量: {len(model.feature_names) if model.feature_names else 'Unknown'}")
            
            # 获取特征列表
            feature_columns = model.feature_names
            if not feature_columns:
                logger.error("模型未包含特征名称信息！")
                return False
        
        # ========== 2. 加载回测数据 ==========
        with LoggerContext(logger, "加载回测数据"):
            loader = DataLoader(
                source=config['data']['source'],
                token=config['data'].get('token')
            )
            
            # 加载股票列表
            stock_list = loader.load_stock_list()
            
            # 确定股票池
            universe = config['data'].get('universe', 'all')
            if universe == 'test':
                stock_codes = stock_list['ts_code'].tolist()[:30]
                logger.info(f"  股票池: 测试池 (前30只)")
            else:
                # 使用所有缓存的股票
                from pathlib import Path
                cache_dir = Path('./data/stock_daily')
                if cache_dir.exists():
                    stock_files = list(cache_dir.glob('*.csv'))
                    stock_codes = [f.stem for f in stock_files]
                    logger.info(f"  股票池: 全部缓存股票 ({len(stock_codes)} 只)")
                else:
                    stock_codes = stock_list['ts_code'].tolist()[:100]
                    logger.info(f"  股票池: 默认 (前100只)")
            
            # 加载回测期间的数据
            logger.info("加载价格数据...")
            price_data = loader.load_daily_data(
                start_date=config['backtest']['start_date'],
                end_date=config['backtest']['end_date'],
                stock_codes=stock_codes
            )
            
            if len(price_data) == 0:
                logger.error("未加载到价格数据！")
                return False
            
            logger.info(f"✓ 价格数据加载完成: {len(price_data)} 条记录")
            logger.info(f"  股票数: {price_data['stock_code'].nunique()}")
            logger.info(f"  日期范围: {price_data['date'].min()} 至 {price_data['date'].max()}")
            
            # 加载因子数据（从缓存读取）
            logger.info("查找因子数据缓存...")
            
            cache_dir = './cache'
            latest_cache = os.path.join(cache_dir, 'factors_latest.parquet')
            
            # 优先使用最新的缓存
            if os.path.exists(latest_cache):
                logger.info(f"  从缓存加载: factors_latest.parquet")
                factor_data = pd.read_parquet(latest_cache)
                logger.info(f"✓ 因子数据加载完成: {len(factor_data)} 条记录")
            else:
                # 查找其他缓存文件
                cache_files = []
                if os.path.exists(cache_dir):
                    cache_files = [f for f in os.listdir(cache_dir) 
                                 if f.startswith('factors_') and f.endswith('.parquet')]
                
                if cache_files:
                    # 使用最新的缓存
                    cache_files.sort(reverse=True)
                    cache_path = os.path.join(cache_dir, cache_files[0])
                    logger.info(f"  从缓存加载: {cache_files[0]}")
                    
                    factor_data = pd.read_parquet(cache_path)
                    logger.info(f"✓ 因子数据加载完成: {len(factor_data)} 条记录")
                else:
                    logger.error("未找到因子数据缓存！")
                    logger.info("请先运行 main.py 生成因子数据")
                    logger.info("或者使用以下命令:")
                    logger.info("  python main.py  # 完整流程，会自动保存因子数据")
                    return False
            
            # 过滤回测期间的数据
            backtest_data = factor_data[
                (factor_data['date'] >= config['backtest']['start_date']) &
                (factor_data['date'] <= config['backtest']['end_date'])
            ].copy()
            
            if len(backtest_data) == 0:
                logger.error("回测期间无数据！")
                return False
            
            logger.info(f"✓ 回测数据准备完成: {len(backtest_data)} 条记录")
            
            # 确保包含所有需要的特征
            missing_features = set(feature_columns) - set(backtest_data.columns)
            if missing_features:
                logger.warning(f"缺失特征: {missing_features}")
                logger.info("尝试补充缺失特征...")
                for feat in missing_features:
                    backtest_data[feat] = 0
        
        # ========== 3. 初始化回测组件 ==========
        with LoggerContext(logger, "初始化回测组件"):
            # 选股器
            selector = StockSelector(
                model=model,
                method=config['backtest']['selection_method'],
                top_n=config['backtest']['top_n'],
                weight_method=config['backtest']['weight_method']
            )
            
            # 组合管理器
            portfolio = PortfolioManager(
                initial_capital=config['backtest']['initial_capital'],
                commission_rate=config['backtest']['commission_rate'],
                slippage=config['backtest']['slippage']
            )
            
            # 回测器
            backtester = Backtester(
                model=model,
                stock_selector=selector,
                portfolio_manager=portfolio,
                data=backtest_data,
                feature_columns=feature_columns,
                label_col='label'  # 假设有label列
            )
            
            logger.info("✓ 回测组件初始化完成")
        
        # ========== 4. 运行回测 ==========
        with LoggerContext(logger, "运行回测"):
            results = backtester.run_backtest(
                start_date=config['backtest']['start_date'],
                end_date=config['backtest']['end_date'],
                rebalance_freq=config['backtest']['rebalance_freq'],
                train_period=252,
                use_rolling_train=False,  # 不重新训练
                retrain_freq=60
            )
            
            if len(results) > 0:
                logger.info(f"✓ 回测完成: {len(results)} 个调仓期")
            else:
                logger.error("回测失败！")
                return False
        
        # ========== 5. 绩效评估 ==========
        with LoggerContext(logger, "绩效评估"):
            daily_values = backtester.portfolio.get_daily_values_df()
            
            if len(daily_values) > 0:
                # 构建净值序列
                portfolio_values = pd.Series(
                    daily_values['portfolio_value'].values,
                    index=pd.to_datetime(daily_values['date'])
                )
                
                # 加载基准数据
                benchmark_values = None
                if config['backtest'].get('benchmark'):
                    try:
                        benchmark_code = config['backtest']['benchmark']
                        benchmark_data = loader.load_index_data(
                            index_code=benchmark_code,
                            start_date=config['backtest']['start_date'],
                            end_date=config['backtest']['end_date']
                        )
                        
                        if not benchmark_data.empty:
                            benchmark_data = benchmark_data.set_index('date')
                            benchmark_data.index = pd.to_datetime(benchmark_data.index)
                            
                            first_date = portfolio_values.index[0]
                            if first_date in benchmark_data.index:
                                initial_close = benchmark_data.loc[first_date, 'close']
                                benchmark_values = benchmark_data['close'] / initial_close * config['backtest']['initial_capital']
                                
                                # 只保留交易日数据
                                common_dates = portfolio_values.index.intersection(benchmark_values.index)
                                benchmark_values = benchmark_values[common_dates]
                            
                            logger.info(f"  基准数据: {benchmark_code}, {len(benchmark_data)} 条")
                    except Exception as e:
                        logger.warning(f"  加载基准数据失败: {e}")
                        benchmark_values = None
                
                # 创建评估器
                evaluator = PerformanceEvaluator(
                    portfolio_values=portfolio_values,
                    benchmark_values=benchmark_values,
                    risk_free_rate=config['backtest'].get('risk_free_rate', 0.03)
                )
                
                # 计算指标
                metrics = evaluator.calculate_all_metrics()
                
                # 生成报告
                report = evaluator.generate_report()
                print("\n" + report)
                
                # 保存指标
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                metrics_df = pd.DataFrame([metrics])
                metrics_path = f"{config['output']['report_path']}/backtest_metrics_{timestamp}.csv"
                metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
                logger.info(f"  绩效指标已保存: {metrics_path}")
                
                # 保存报告
                report_path = f"{config['output']['report_path']}/backtest_report_{timestamp}.txt"
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                logger.info(f"  绩效报告已保存: {report_path}")
                
                logger.info("✓ 绩效评估完成")
        
        # ========== 6. 生成图表 ==========
        if config['output'].get('generate_plots', True):
            with LoggerContext(logger, "生成图表"):
                daily_values = backtester.portfolio.get_daily_values_df()
                
                if len(daily_values) > 0:
                    portfolio_values = pd.Series(
                        daily_values['portfolio_value'].values,
                        index=pd.to_datetime(daily_values['date'])
                    )
                    
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    figure_path = config['output']['figure_path']
                    
                    # 净值曲线
                    logger.info("  生成净值曲线...")
                    plot_equity_curve(
                        portfolio_values=portfolio_values,
                        benchmark_values=benchmark_values if 'benchmark_values' in locals() else None,
                        title='回测净值曲线（使用已保存模型）',
                        save_path=f"{figure_path}/backtest_equity_{timestamp}.png"
                    )
                    
                    # 回撤曲线
                    logger.info("  生成回撤曲线...")
                    plot_drawdown(
                        portfolio_values=portfolio_values,
                        benchmark_values=benchmark_values if 'benchmark_values' in locals() else None,
                        title='回测回撤分析',
                        save_path=f"{figure_path}/backtest_drawdown_{timestamp}.png"
                    )
                    
                    # 收益率分布
                    logger.info("  生成收益率分布...")
                    plot_returns_distribution(
                        portfolio_values=portfolio_values,
                        benchmark_values=benchmark_values if 'benchmark_values' in locals() else None,
                        save_path=f"{figure_path}/backtest_returns_{timestamp}.png"
                    )
                    
                    # 综合看板
                    logger.info("  生成综合看板...")
                    create_dashboard(
                        portfolio_values=portfolio_values,
                        benchmark_values=benchmark_values if 'benchmark_values' in locals() else None,
                        metrics=metrics if 'metrics' in locals() else None,
                        save_path=f"{figure_path}/backtest_dashboard_{timestamp}.png"
                    )
                    
                    logger.info(f"✓ 图表已保存至: {figure_path}")
        
        logger.info("="*60)
        logger.info("回测完成！")
        logger.info("="*60)
        return True
        
    except Exception as e:
        logger.error(f"回测失败: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    # 从命令行参数获取模型路径（可选）
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        print(f"使用指定模型: {model_path}")
        success = main(model_path=model_path)
    else:
        # 使用最新的模型
        print("使用最新的模型文件")
        success = main()
    
    if success:
        print("\n✓ 回测成功完成！")
    else:
        print("\n✗ 回测失败！")
        sys.exit(1)
