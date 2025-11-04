"""
XGBoost多因子选股 - 完整主程序
整合所有模块，实现从数据加载到回测评估的完整流程
"""

import yaml
import json
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import os
warnings.filterwarnings('ignore')

# 导入项目模块
from src.data.data_loader import DataLoader
from src.data.data_processor import DataProcessor
from src.factors.factor_calculator import FactorCalculator
from src.factors.factor_processor import FactorProcessor
from src.factors.factor_selector import FactorSelector
from src.model.label_builder import LabelBuilder
from src.model.model_tuner import ModelTuner
from src.model.xgb_model import XGBoostModel
from src.backtest.stock_selector import StockSelector
from src.backtest.portfolio_manager import PortfolioManager
from src.backtest.backtester import Backtester
from src.backtest.evaluator import PerformanceEvaluator
from src.utils.logger import setup_logger, LoggerContext
from src.utils.visualization import (
    plot_equity_curve, 
    plot_drawdown, 
    plot_returns_distribution,
    plot_rolling_metrics,
    plot_monthly_returns,
    plot_feature_importance,
    create_dashboard
)


def load_config(config_path='config/config.yaml'):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """主函数 - 完整流程"""
    
    # ========== 初始化 ==========
    print("\n" + "="*80)
    print("XGBoost多因子选股系统")
    print("="*80)
    
    # 加载配置
    config = load_config()
    
    # 设置日志
    logger = setup_logger(
        name='xgboost_stock',
        log_dir=config['output']['log_path'],
        level=20  # INFO
    )
    
    logger.info("="*60)
    logger.info("系统启动")
    logger.info("="*60)
    
    # 创建输出目录
    for path in [config['output']['model_path'], 
                 config['output']['report_path'],
                 config['output']['figure_path']]:
        os.makedirs(path, exist_ok=True)
    
    try:
        # ========== 1. 数据加载 ==========
        with LoggerContext(logger, "数据加载"):
            logger.info("数据源: %s", config['data']['source'])
            logger.info("数据范围: %s 至 %s", 
                       config['data']['start_date'], 
                       config['data']['end_date'])
            
            # 初始化数据加载器
            loader = DataLoader(
                source=config['data']['source'],
                token=config['data'].get('token')
            )
            
            # 加载股票列表
            logger.info("加载股票列表...")
            stock_list = loader.load_stock_list()
            logger.info(f"  获取到 {len(stock_list)} 只股票")
            
            # 根据配置选择股票池
            universe = config['data'].get('universe', 'all')
            if universe == 'hs300':
                # TODO: 获取沪深300成分股
                stock_codes = stock_list['ts_code'].tolist()[:50]
                logger.info(f"  股票池: 沪深300 (暂用前50只)")
            elif universe == 'zz500':
                stock_codes = stock_list['ts_code'].tolist()[:100]
                logger.info(f"  股票池: 中证500 (前100只)")
            elif universe == 'test':
                # 测试股票池，使用30只确保快速测试
                stock_codes = stock_list['ts_code'].tolist()[:30]
                logger.info(f"  股票池: 测试池 (前30只)")
            else:
                # 使用所有缓存的股票（从缓存目录获取）
                from pathlib import Path
                cache_dir = Path('./data/stock_daily')
                if cache_dir.exists():
                    stock_files = list(cache_dir.glob('*.csv'))
                    stock_codes = [f.stem for f in stock_files]
                    logger.info(f"  股票池: 全部缓存股票 ({len(stock_codes)} 只)")
                else:
                    # 如果缓存目录不存在，使用前100只
                    stock_codes = stock_list['ts_code'].tolist()[:100]
                    logger.info(f"  股票池: 默认股票池 (前100只)")
            
            # 加载日线数据
            logger.info("加载日线数据...")
            price_data = loader.load_daily_data(
                start_date=config['data']['start_date'],
                end_date=config['data']['end_date'],
                stock_codes=stock_codes
            )
            
            if len(price_data) > 0:
                logger.info(f"✓ 数据加载完成: {len(price_data)} 条记录")
                logger.info(f"  股票数: {price_data['stock_code'].nunique()}")
                logger.info(f"  日期范围: {price_data['date'].min()} 至 {price_data['date'].max()}")
            else:
                logger.error("未加载到任何数据！")
                return False
            
            # 加载市值和估值数据
            logger.info("加载市值和估值数据...")
            try:
                # 获取所有交易日期
                trade_dates = price_data['date'].unique()
                market_data_list = []
                
                # 每隔5个交易日采样一次，减少API调用
                sample_dates = trade_dates[::5]
                
                for i, date in enumerate(sample_dates):
                    if i > 0 and i % 10 == 0:
                        logger.info(f"  已加载 {i}/{len(sample_dates)} 个交易日的市值数据")
                    
                    # 转换pandas Timestamp为字符串
                    if hasattr(date, 'strftime'):
                        date_str = date.strftime('%Y%m%d')
                    else:
                        date_str = pd.Timestamp(date).strftime('%Y%m%d')
                    
                    try:
                        market_df = loader.load_market_data(date_str)
                        if len(market_df) > 0:
                            market_data_list.append(market_df)
                        
                        # API限流，增加等待时间
                        import time
                        time.sleep(0.5)
                    except Exception as e:
                        error_msg = str(e)
                        if "每分钟最多访问" in error_msg or "访问频率" in error_msg:
                            logger.warning(f"  触发API频率限制，等待60秒...")
                            import time
                            time.sleep(60)
                            # 重试一次
                            try:
                                market_df = loader.load_market_data(date_str)
                                if len(market_df) > 0:
                                    market_data_list.append(market_df)
                            except:
                                pass
                        logger.warning(f"  跳过日期 {date_str}: {error_msg}")
                        continue
                
                if market_data_list:
                    market_data = pd.concat(market_data_list, ignore_index=True)
                    logger.info(f"✓ 市值数据加载完成: {len(market_data)} 条记录")
                    
                    # 合并价格数据和市值数据
                    logger.info("合并价格和市值数据...")
                    price_data = pd.merge(
                        price_data,
                        market_data[['stock_code', 'date', 'pe', 'pb', 'ps', 'market_cap', 'float_market_cap', 'turnover_rate']],
                        on=['stock_code', 'date'],
                        how='left'
                    )
                    logger.info(f"✓ 数据合并完成: {price_data.shape}")
                else:
                    logger.warning("未能加载市值数据，将仅使用价格数据")
                    
            except Exception as e:
                logger.warning(f"加载市值数据失败: {str(e)}")
                logger.info("将继续使用纯价格数据")
            
            # 加载行业分类数据
            logger.info("加载行业分类数据...")
            try:
                industry_data = loader.load_industry_data(stock_codes=stock_codes)
                if len(industry_data) > 0:
                    logger.info(f"✓ 行业数据加载完成: {len(industry_data)} 只股票")
                    logger.info(f"  行业数量: {industry_data['industry'].nunique()}")
                    
                    # 合并行业数据（按stock_code合并，不需要date）
                    price_data = pd.merge(
                        price_data,
                        industry_data[['stock_code', 'industry', 'industry_code']],
                        on='stock_code',
                        how='left'
                    )
                    logger.info(f"✓ 行业数据合并完成")
                else:
                    logger.warning("未能加载行业数据")
            except Exception as e:
                logger.warning(f"加载行业数据失败: {str(e)}")
            
            # 加载财务指标数据
            logger.info("加载财务指标数据...")
            try:
                financial_data = loader.load_financial_indicators(
                    start_date=config['data']['start_date'],
                    end_date=config['data']['end_date'],
                    stock_codes=stock_codes
                )
                if len(financial_data) > 0:
                    logger.info(f"✓ 财务指标加载完成: {len(financial_data)} 条记录")
                    
                    # 财务数据需要按最近的报告期匹配
                    # 由于pd.merge_asof在某些pandas版本中可能有排序检测问题
                    # 我们使用更稳健的方法：扩展财务数据到每一天
                    logger.info("  准备合并财务数据...")
                    
                    # 为每个股票的每个财报期创建一个有效期间
                    fin_expanded = []
                    for stock_code in financial_data['stock_code'].unique():
                        stock_fin = financial_data[financial_data['stock_code'] == stock_code].copy()
                        stock_fin = stock_fin.sort_values('end_date')
                        stock_price = price_data[price_data['stock_code'] == stock_code]['date'].unique()
                        
                        for idx, row in stock_fin.iterrows():
                            # 找到下一个财报日期
                            next_dates = stock_fin[stock_fin['end_date'] > row['end_date']]['end_date']
                            if len(next_dates) > 0:
                                valid_until = next_dates.iloc[0]
                            else:
                                valid_until = stock_price.max()
                            
                            # 为这个财报期间的所有交易日添加财务数据
                            valid_dates = stock_price[(stock_price > row['end_date']) & (stock_price <= valid_until)]
                            for date in valid_dates:
                                fin_row = row.to_dict()
                                fin_row['date'] = date
                                fin_expanded.append(fin_row)
                    
                    if fin_expanded:
                        fin_expanded_df = pd.DataFrame(fin_expanded)
                        # 移除end_date列，保留date列
                        fin_expanded_df = fin_expanded_df.drop(columns=['end_date'])
                        
                        logger.info(f"  财务数据扩展到 {len(fin_expanded_df)} 条日度记录")
                        
                        # 使用普通merge合并
                        price_data = pd.merge(
                            price_data,
                            fin_expanded_df,
                            on=['stock_code', 'date'],
                            how='left'
                        )
                        logger.info(f"✓ 财务数据合并完成: {price_data.shape}")
                    else:
                        logger.warning("  财务数据扩展失败")
                else:
                    logger.warning("未能加载财务指标数据")
            except Exception as e:
                logger.warning(f"加载财务指标失败: {str(e)}")
                import traceback
                logger.warning(f"  详细错误: {traceback.format_exc()}")
            
            # 加载宏观经济数据
            logger.info("加载宏观经济数据...")
            try:
                macro_data = loader.load_macro_data(
                    start_date=config['data']['start_date'],
                    end_date=config['data']['end_date']
                )
                if len(macro_data) > 0:
                    logger.info(f"✓ 宏观数据加载完成: {len(macro_data)} 条记录")
                    
                    # 显示加载的宏观指标
                    macro_cols = [col for col in macro_data.columns if col != 'date']
                    logger.info(f"  宏观指标: {', '.join(macro_cols[:10])}")
                    
                    # 宏观数据按日期合并（广播到每只股票）
                    price_data = pd.merge(
                        price_data,
                        macro_data,
                        on='date',
                        how='left'
                    )
                    logger.info(f"✓ 宏观数据合并完成: {price_data.shape}")
                else:
                    logger.warning("未能加载宏观数据")
            except Exception as e:
                logger.warning(f"加载宏观数据失败: {str(e)}")
        
        # ========== 2. 数据处理 ==========
        with LoggerContext(logger, "数据处理"):
            logger.info("初始化数据处理器...")
            processor = DataProcessor()
            
            # 数据清洗
            logger.info("执行数据清洗...")
            cleaned_data = processor.clean_data(
                price_data,
                drop_st=config['data']['cleaning'].get('drop_st', True),
                drop_suspended=config['data']['cleaning'].get('drop_suspended', True),
                min_liquidity=config['data']['cleaning'].get('min_liquidity', 1000000),
                min_price=config['data']['cleaning'].get('min_price', 1.0),
                max_price=config['data']['cleaning'].get('max_price', 1000.0)
            )
            
            if len(cleaned_data) == 0:
                logger.error("数据清洗后无剩余数据！")
                return False
            
            # 处理缺失值
            logger.info("处理缺失值...")
            numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
            if 'date' in cleaned_data.columns:
                numeric_columns = [col for col in numeric_columns if col not in ['date']]
            
            processed_data = processor.handle_missing_values(
                cleaned_data,
                columns=numeric_columns[:5],  # 只处理前5列作为示例
                method='forward_fill'
            )
            
            # 数据统计
            summary = processor.get_data_summary(processed_data)
            logger.info(f"数据处理完成:")
            logger.info(f"  总记录数: {summary['total_rows']}")
            logger.info(f"  股票数量: {summary['unique_stocks']}")
            logger.info(f"  日期范围: {summary['date_range']}")
            logger.info(f"  内存占用: {summary['memory_usage']}")
            
            logger.info("✓ 数据处理完成")
        
        # ========== 3. 因子计算 ==========
        with LoggerContext(logger, "因子计算"):
            logger.info("初始化因子计算器...")
            logger.info(f"输入数据: {len(processed_data)} 条记录, {processed_data['stock_code'].nunique()} 只股票")
            
            # 初始化因子计算器
            calc = FactorCalculator(processed_data)
            
            # 计算所有因子
            logger.info("计算技术指标因子...")
            factor_data = calc.calculate_all_factors()
            
            # 统计因子信息
            original_cols = ['stock_code', 'date', 'open', 'high', 'low', 'close', 
                           'pre_close', 'change', 'pct_chg', 'vol', 'amount']
            factor_cols = [col for col in factor_data.columns if col not in original_cols]
            
            logger.info(f"✓ 因子计算完成")
            logger.info(f"  计算因子数: {len(factor_cols)}")
            logger.info(f"  因子列表: {factor_cols[:10]}..." if len(factor_cols) > 10 else f"  因子列表: {factor_cols}")
            logger.info(f"  数据形状: {factor_data.shape}")
        
        # ========== 4. 因子预处理 ==========
        with LoggerContext(logger, "因子预处理"):
            logger.info("初始化因子预处理器...")
            
            # 识别因子列（排除基础数据列）
            base_cols = ['stock_code', 'date', 'open', 'high', 'low', 'close', 
                        'pre_close', 'change', 'pct_chg', 'vol', 'amount',
                        'volume', 'ts_code', 'trade_date']
            
            # 获取所有因子列
            factor_cols = [col for col in factor_data.columns 
                          if col not in base_cols and factor_data[col].dtype in ['float64', 'int64']]
            
            logger.info(f"识别到 {len(factor_cols)} 个因子待处理")
            logger.info(f"因子列表样例: {factor_cols[:5]}...")
            
            # 初始化因子预处理器
            from src.factors.factor_processor import FactorProcessor
            factor_processor = FactorProcessor(factor_data, factor_cols)
            
            # 执行预处理流程
            logger.info("执行因子预处理流程:")
            logger.info(f"  - 去极值方法: {config['factors']['preprocessing']['winsorize_method']}")
            logger.info(f"  - 标准化方法: {config['factors']['preprocessing']['standardize_method']}")
            logger.info(f"  - 中性化: {config['factors']['preprocessing']['neutralize']}")
            logger.info(f"  - 缺失值填充: {config['factors']['preprocessing']['fill_method']}")
            
            processed_factor_data = factor_processor.process_pipeline(
                winsorize_method=config['factors']['preprocessing']['winsorize_method'],
                standardize_method=config['factors']['preprocessing']['standardize_method'],
                neutralize=config['factors']['preprocessing']['neutralize'],
                fill_method=config['factors']['preprocessing']['fill_method']
            )
            
            # 统计预处理效果
            logger.info("✓ 因子预处理完成")
            logger.info(f"  处理因子数: {len(factor_cols)}")
            logger.info(f"  数据形状: {processed_factor_data.shape}")
            
            # 检查缺失值
            missing_counts = processed_factor_data[factor_cols].isna().sum()
            total_missing = missing_counts.sum()
            if total_missing > 0:
                logger.warning(f"  剩余缺失值: {total_missing} 个")
                top_missing = missing_counts[missing_counts > 0].sort_values(ascending=False).head(5)
                for col, count in top_missing.items():
                    logger.warning(f"    {col}: {count}")
            else:
                logger.info(f"  ✓ 无缺失值")
            
            logger.info("✓ 因子预处理完成")
        
        # ========== 5. 因子筛选 ==========
        with LoggerContext(logger, "因子筛选"):
            # 注意：因子筛选需要先构建标签(forward_return)用于IC计算
            logger.info("构建标签用于因子IC计算...")
            from src.model.label_builder import LabelBuilder
            
            temp_label_builder = LabelBuilder(processed_factor_data)
            forward_days = config['label']['forward_days']
            label_type = config['label']['type']
            
            # 创建临时标签数据用于IC计算
            temp_labeled_data = temp_label_builder.create_return_label(
                forward_days=forward_days,
                label_type=label_type,
                threshold=0.0,
                quantiles=config['label'].get('quantiles', [0.3, 0.7]) if label_type == 'classification' else None
            )
            logger.info(f"  临时标签数据: {len(temp_labeled_data)} 条记录")
            
            if not config['factors']['selection']['use_selection']:
                logger.info("因子筛选已禁用，使用所有因子")
                selected_factors = factor_cols
            else:
                logger.info("初始化因子筛选器...")
                from src.factors.factor_selector import FactorSelector
                
                selector = FactorSelector(
                    factor_data=temp_labeled_data,
                    factor_columns=factor_cols,
                    forward_return_col='forward_return'
                )
                
                # 计算IC
                logger.info("计算因子IC值...")
                ic_df = selector.calculate_ic(method='spearman')
                
                # 评估因子
                logger.info("评估因子表现...")
                evaluation = selector.evaluate_factors(ic_df)
                
                # 显示Top10因子
                top_factors = evaluation.head(10)
                logger.info("Top 10 因子 (按ICIR排序):")
                for idx, row in top_factors.iterrows():
                    logger.info(f"  {row['factor']}: IC={row['IC_mean']:.4f}, ICIR={row['ICIR']:.4f}, 胜率={row['IC_win_rate']:.2%}")
                
                # 基于IC筛选因子 - 优化策略
                ic_threshold = config['factors']['selection']['ic_threshold']
                icir_threshold = config['factors']['selection']['icir_threshold']
                min_factors = config['factors']['selection'].get('min_factors', 10)
                max_factors = config['factors']['selection'].get('max_factors', 30)
                
                logger.info(f"筛选策略:")
                logger.info(f"  - IC绝对值阈值: >{ic_threshold}")
                logger.info(f"  - ICIR阈值: >{icir_threshold}")
                logger.info(f"  - 保留因子范围: {min_factors}-{max_factors}个")
                
                # 先按IC筛选
                selected_by_ic = selector.select_by_ic(
                    ic_threshold=ic_threshold,
                    icir_threshold=icir_threshold,
                    win_rate_threshold=0.45  # 降低胜率要求
                )
                
                logger.info(f"IC筛选后: {len(selected_by_ic)} 个因子")
                
                # 如果筛选出的因子太少，降低阈值
                if len(selected_by_ic) < min_factors:
                    logger.warning(f"筛选因子数({len(selected_by_ic)})少于最小值({min_factors})，降低阈值重新筛选...")
                    
                    # 直接选择ICIR最高的因子
                    selected_by_ic = evaluation.head(max_factors)['factor'].tolist()
                    logger.info(f"调整后: 选择Top {len(selected_by_ic)} ICIR因子")
                
                # 如果筛选出的因子太多，只保留最优的
                if len(selected_by_ic) > max_factors:
                    logger.info(f"筛选因子数({len(selected_by_ic)})超过最大值({max_factors})，保留最优因子...")
                    # 按ICIR排序，保留前N个
                    top_n = evaluation[evaluation['factor'].isin(selected_by_ic)].head(max_factors)
                    selected_by_ic = top_n['factor'].tolist()
                    logger.info(f"调整后: {len(selected_by_ic)} 个因子")
                
                # 去除高相关因子
                if len(selected_by_ic) > 1:
                    corr_threshold = config['factors']['selection']['correlation_threshold']
                    logger.info(f"去除相关性>{corr_threshold}的冗余因子...")
                    
                    selected_factors = selector.remove_correlated_factors(
                        correlation_threshold=corr_threshold,
                        factors=selected_by_ic
                    )
                    
                    logger.info(f"去相关后: {len(selected_factors)} 个因子")
                    
                    # 确保最少保留min_factors个因子
                    if len(selected_factors) < min_factors and len(selected_by_ic) >= min_factors:
                        logger.warning(f"去相关后因子过少，调整相关性阈值...")
                        # 提高相关性阈值，保留更多因子
                        selected_factors = selector.remove_correlated_factors(
                            correlation_threshold=0.9,  # 提高阈值
                            factors=selected_by_ic
                        )
                        logger.info(f"调整后: {len(selected_factors)} 个因子")
                else:
                    selected_factors = selected_by_ic
                
                # 显示最终选择的因子及其评估指标
                logger.info(f"✓ 因子筛选完成")
                logger.info(f"  最终因子数: {len(selected_factors)}")
                logger.info(f"  筛选率: {len(selected_factors)/len(factor_cols)*100:.1f}%")
                
                # 显示选中因子的详细信息
                selected_eval = evaluation[evaluation['factor'].isin(selected_factors)].sort_values('ICIR', ascending=False)
                logger.info(f"\n最终选中因子详情:")
                for idx, row in selected_eval.iterrows():
                    logger.info(f"  • {row['factor']}: IC={row['IC_mean']:.4f}, ICIR={row['ICIR']:.4f}, 胜率={row['IC_win_rate']:.2%}")
            
            # ic_df = selector.calculate_ic()
            # selected_factors = selector.select_by_ic(...)
            # final_factors = selector.remove_correlated_factors(...)
            
            logger.info("✓ 因子筛选完成")
        
        # ========== 6. 标签构建 ==========
        with LoggerContext(logger, "标签构建"):
            logger.info("基于筛选后的因子构建最终标签...")
            from src.model.label_builder import LabelBuilder
            
            # 只保留选中的因子列
            selected_cols = ['stock_code', 'date', 'open', 'high', 'low', 'close', 
                           'pre_close', 'change', 'pct_chg', 'vol', 'amount'] + selected_factors
            # 确保所有列都存在
            available_cols = [col for col in selected_cols if col in processed_factor_data.columns]
            final_factor_data = processed_factor_data[available_cols].copy()
            
            logger.info(f"  选中因子数: {len(selected_factors)}")
            logger.info(f"  最终特征列数: {len(available_cols)}")
            
            # 构建标签
            label_builder = LabelBuilder(final_factor_data)
            forward_days = config['label']['forward_days']
            label_type = config['label']['type']
            
            logger.info(f"构建标签:")
            logger.info(f"  - 前瞻天数: {forward_days}")
            logger.info(f"  - 标签类型: {label_type}")
            
            labeled_data = label_builder.create_return_label(
                forward_days=forward_days,
                label_type=label_type,
                threshold=0.0,
                quantiles=config['label'].get('quantiles', [0.3, 0.7]) if label_type == 'classification' else None
            )
            
            logger.info(f"✓ 标签构建完成")
            logger.info(f"  最终样本数: {len(labeled_data)}")
            if 'label' in labeled_data.columns:
                logger.info(f"  标签分布: {labeled_data['label'].value_counts().to_dict()}")
            
        # ========== 7. 模型训练 ==========
        with LoggerContext(logger, "模型训练"):
            # 准备训练数据
            logger.info("准备训练数据...")
            
            # 分离特征和标签
            feature_cols = [col for col in labeled_data.columns 
                           if col not in ['stock_code', 'date', 'label', 'forward_return']]
            
            X = labeled_data[feature_cols].copy()
            y = labeled_data['label'].copy()
            
            logger.info(f"  特征数: {len(feature_cols)}")
            logger.info(f"  样本数: {len(X)}")
            logger.info(f"  标签分布: {y.value_counts().to_dict()}")
            
            # 时间序列切分（按日期）
            dates = labeled_data['date'].unique()
            dates = sorted(dates)
            split_idx = int(len(dates) * (1 - config['model']['training']['validation_split']))
            train_dates = dates[:split_idx]
            val_dates = dates[split_idx:]
            
            train_mask = labeled_data['date'].isin(train_dates)
            val_mask = labeled_data['date'].isin(val_dates)
            
            X_train = X[train_mask]
            y_train = y[train_mask]
            X_val = X[val_mask]
            y_val = y[val_mask]
            
            logger.info(f"  训练集: {len(X_train)} 样本, {len(train_dates)} 天")
            logger.info(f"  验证集: {len(X_val)} 样本, {len(val_dates)} 天")
            
            # 如果需要调参
            if config['model'].get('tuning', {}).get('use_tuning', False):
                logger.info("开始超参数优化...")
                
                # 检测类别数
                num_class = None
                if config['model']['task_type'] == 'classification':
                    num_class = int(y_train.nunique())
                
                from src.model.model_tuner import ModelTuner
                tuner = ModelTuner(
                    task_type=config['model']['task_type'],
                    cv_splits=config['model']['tuning']['cv_splits'],
                    num_class=num_class
                )
                
                method = config['model']['tuning']['method']
                if method == 'grid_search':
                    best_params = tuner.grid_search(X_train, y_train)
                else:
                    best_params = tuner.random_search(X_train, y_train, n_iter=20)
                
                logger.info(f"  最优参数: {best_params}")
                model_params = best_params
            else:
                # 使用配置的参数
                model_params = config['model']['params']
                logger.info("使用配置的模型参数")
            
            # 初始化模型
            logger.info("初始化模型...")
            
            # 检测类别数（用于多分类）
            num_class = None
            if config['model']['task_type'] == 'classification':
                num_class = int(y_train.nunique())
                logger.info(f"  分类类别数: {num_class}")
            
            model = XGBoostModel(
                task_type=config['model']['task_type'],
                params=model_params,
                num_class=num_class
            )
            
            # 训练模型
            logger.info("开始训练模型...")
            use_validation = config['model']['training']['use_validation']
            early_stopping = config['model']['training']['early_stopping_rounds']
            verbose = config['model']['training']['verbose']
            
            if use_validation and len(X_val) > 0:
                model.train(
                    X_train, y_train,
                    X_val, y_val,
                    early_stopping_rounds=early_stopping,
                    verbose=verbose
                )
            else:
                model.train(X_train, y_train, verbose=verbose)
            
            # 显示特征重要性
            logger.info("\nTop 10 重要特征:")
            top_features = model.get_feature_importance(top_n=10)
            for idx, row in top_features.iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            # 保存模型
            model_path = f"{config['output']['model_path']}/model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            model.save_model(model_path)
            logger.info(f"模型已保存至: {model_path}")
            
            logger.info("✓ 模型训练完成")
        
        # ========== 8. 回测 ==========
        with LoggerContext(logger, "回测"):
            logger.info("初始化回测组件...")
            
            # 初始化选股器
            from src.backtest.stock_selector import StockSelector
            selector = StockSelector(
                model=model,
                n_stocks=config['strategy']['n_stocks'],
                method=config['strategy']['method']
            )
            logger.info(f"  选股策略: {config['strategy']['method']}")
            logger.info(f"  持仓数量: {config['strategy']['n_stocks']}")
            
            # 初始化组合管理器
            from src.backtest.portfolio_manager import PortfolioManager
            portfolio = PortfolioManager(
                initial_capital=config['backtest']['initial_capital'],
                commission_rate=config['backtest']['commission_rate'],
                slippage=config['backtest']['slippage']
            )
            logger.info(f"  初始资金: {config['backtest']['initial_capital']:,.0f}")
            
            # 初始化回测引擎
            from src.backtest.backtester import Backtester
            backtester = Backtester(
                model=model,
                stock_selector=selector,
                portfolio_manager=portfolio,
                data=labeled_data,
                feature_columns=feature_cols,
                label_col='label'
            )
            
            # 运行回测
            logger.info("开始回测...")
            backtest_results = backtester.run_backtest(
                start_date=config['backtest']['start_date'],
                end_date=config['backtest']['end_date'],
                rebalance_freq=config['backtest']['rebalance_freq'],
                train_period=config['backtest']['train_period'],
                use_rolling_train=config['backtest']['use_rolling_train'],
                retrain_freq=config['backtest']['retrain_freq']
            )
            
            # 获取回测摘要
            backtest_summary = backtester.get_summary()
            logger.info("\n回测摘要:")
            logger.info(f"  回测期间: {backtest_summary.get('start_date')} - {backtest_summary.get('end_date')}")
            logger.info(f"  交易天数: {backtest_summary.get('trading_days')}")
            logger.info(f"  调仓次数: {backtest_summary.get('n_rebalances')}")
            logger.info(f"  总收益率: {backtest_summary.get('total_return', 0):.2f}%")
            logger.info(f"  最终市值: {backtest_summary.get('final_value', 0):,.0f}")
            
            logger.info("✓ 回测完成")
        
        # ========== 9. 绩效评估 ==========
        with LoggerContext(logger, "绩效评估"):
            from src.backtest.evaluator import PerformanceEvaluator
            
            # 获取组合净值序列
            daily_values = backtester.portfolio.get_daily_values_df()
            
            if len(daily_values) > 0:
                # 构建净值序列
                portfolio_values = pd.Series(
                    daily_values['portfolio_value'].values,
                    index=pd.to_datetime(daily_values['date'])
                )
                
                # 加载基准数据（可选）
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
                            # 对齐日期
                            benchmark_data = benchmark_data.set_index('date')
                            benchmark_data.index = pd.to_datetime(benchmark_data.index)
                            
                            # 归一化基准净值
                            first_date = portfolio_values.index[0]
                            if first_date in benchmark_data.index:
                                initial_close = benchmark_data.loc[first_date, 'close']
                                benchmark_values = benchmark_data['close'] / initial_close * config['backtest']['initial_capital']
                                
                                # 只保留组合存在的日期
                                benchmark_values = benchmark_values.reindex(portfolio_values.index).fillna(method='ffill')
                            
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
                
                # 计算所有指标
                metrics = evaluator.calculate_all_metrics()
                
                # 生成报告
                report = evaluator.generate_report()
                print(report)
                
                # 保存指标到文件
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                metrics_df = pd.DataFrame([metrics])
                metrics_path = f"{config['output']['report_path']}/performance_metrics_{timestamp}.csv"
                metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
                logger.info(f"  绩效指标已保存: {metrics_path}")
                
                # 保存报告到文件
                report_path = f"{config['output']['report_path']}/performance_report_{timestamp}.txt"
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                logger.info(f"  绩效报告已保存: {report_path}")
                
                logger.info("✓ 绩效评估完成")
            else:
                logger.warning("  无回测数据，跳过绩效评估")
        
        # ========== 10. 生成图表 ==========
        if config['output'].get('generate_plots', True):
            with LoggerContext(logger, "生成图表"):
                try:
                    # 获取数据
                    daily_values = backtester.portfolio.get_daily_values_df()
                    
                    if len(daily_values) > 0:
                        # 构建净值序列
                        portfolio_values = pd.Series(
                            daily_values['portfolio_value'].values,
                            index=pd.to_datetime(daily_values['date'])
                        )
                        
                        # 如果有基准数据，使用之前加载的
                        benchmark_values_plot = None
                        if 'benchmark_values' in locals() and benchmark_values is not None:
                            benchmark_values_plot = benchmark_values
                        
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        figure_path = config['output']['figure_path']
                        
                        # 1. 净值曲线
                        logger.info("  生成净值曲线...")
                        plot_equity_curve(
                            portfolio_values=portfolio_values,
                            benchmark_values=benchmark_values_plot,
                            title='策略净值曲线',
                            save_path=f"{figure_path}/equity_curve_{timestamp}.png"
                        )
                        
                        # 2. 回撤曲线
                        logger.info("  生成回撤曲线...")
                        plot_drawdown(
                            portfolio_values=portfolio_values,
                            benchmark_values=benchmark_values_plot,
                            title='策略回撤分析',
                            save_path=f"{figure_path}/drawdown_{timestamp}.png"
                        )
                        
                        # 3. 收益率分布
                        logger.info("  生成收益率分布...")
                        plot_returns_distribution(
                            portfolio_values=portfolio_values,
                            benchmark_values=benchmark_values_plot,
                            save_path=f"{figure_path}/returns_dist_{timestamp}.png"
                        )
                        
                        # 4. 滚动指标
                        if len(portfolio_values) >= 60:
                            logger.info("  生成滚动指标...")
                            plot_rolling_metrics(
                                portfolio_values=portfolio_values,
                                window=min(60, len(portfolio_values) // 2),
                                save_path=f"{figure_path}/rolling_metrics_{timestamp}.png"
                            )
                        
                        # 5. 月度收益
                        if len(portfolio_values) >= 30:
                            logger.info("  生成月度收益...")
                            plot_monthly_returns(
                                portfolio_values=portfolio_values,
                                save_path=f"{figure_path}/monthly_returns_{timestamp}.png"
                            )
                        
                        # 6. 因子重要性
                        if hasattr(model, 'get_feature_importance'):
                            logger.info("  生成因子重要性...")
                            importance_df = model.get_feature_importance()
                            if not importance_df.empty:
                                plot_feature_importance(
                                    importance_df=importance_df,
                                    top_n=min(20, len(importance_df)),
                                    save_path=f"{figure_path}/feature_importance_{timestamp}.png"
                                )
                        
                        # 7. 综合看板
                        logger.info("  生成综合看板...")
                        metrics_plot = None
                        if 'metrics' in locals():
                            metrics_plot = metrics
                        
                        create_dashboard(
                            portfolio_values=portfolio_values,
                            benchmark_values=benchmark_values_plot,
                            metrics=metrics_plot,
                            save_path=f"{figure_path}/dashboard_{timestamp}.png"
                        )
                        
                        logger.info(f"✓ 图表生成完成，保存至: {figure_path}")
                    else:
                        logger.warning("  无数据可用于绘图")
                
                except Exception as e:
                    logger.error(f"  图表生成失败: {e}")
                    import traceback
                    traceback.print_exc()
        
        # ========== 11. 保存结果 ==========
        with LoggerContext(logger, "保存结果"):
            # 保存回测结果
            # results.to_csv(...)
            
            # 保存交易记录
            # portfolio.get_trades_df().to_csv(...)
            
            # 保存选股结果
            # selected_stocks.to_csv(...)
            
            # 保存绩效指标
            # metrics_df = pd.DataFrame([metrics])
            # metrics_df.to_csv(...)
            
            logger.info("✓ 结果保存完成")
        
        # ========== 完成 ==========
        logger.info("="*60)
        logger.info("流程执行成功完成！")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error("流程执行失败: %s", str(e), exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✅ 程序执行成功！")
    else:
        print("\n❌ 程序执行失败，请查看日志。")
