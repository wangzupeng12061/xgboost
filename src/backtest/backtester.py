"""
回测引擎模块
实现完整的滚动训练回测流程
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class Backtester:
    """回测引擎类"""
    
    def __init__(self,
                 model,
                 stock_selector,
                 portfolio_manager,
                 data: pd.DataFrame,
                 feature_columns: List[str],
                 label_col: str = 'label'):
        """
        初始化回测引擎
        
        Args:
            model: 训练好的模型（或模型类）
            stock_selector: 选股器
            portfolio_manager: 组合管理器
            data: 完整数据（包含因子和标签）
            feature_columns: 特征列名
            label_col: 标签列名
        """
        self.model = model
        self.selector = stock_selector
        self.portfolio = portfolio_manager
        self.data = data.sort_values(['date', 'stock_code']).reset_index(drop=True)
        self.feature_columns = feature_columns
        self.label_col = label_col
        
        self.results = []
        self.training_dates = []
        
        print(f"Backtester initialized")
        print(f"  Data: {len(data)} rows")
        print(f"  Features: {len(feature_columns)}")
        print(f"  Date range: {data['date'].min()} to {data['date'].max()}")
    
    def run_backtest(self,
                    start_date: str,
                    end_date: str,
                    rebalance_freq: str = '20D',
                    train_period: int = 252,
                    use_rolling_train: bool = True,
                    retrain_freq: int = 60) -> pd.DataFrame:
        """
        运行回测
        
        Args:
            start_date: 回测开始日期
            end_date: 回测结束日期
            rebalance_freq: 调仓频率 ('1D', '5D', '20D', '60D')
            train_period: 训练窗口（交易日）
            use_rolling_train: 是否使用滚动训练
            retrain_freq: 重新训练频率（交易日）
            
        Returns:
            回测结果DataFrame
        """
        print("\n" + "="*80)
        print("开始回测")
        print("="*80)
        print(f"回测期间: {start_date} to {end_date}")
        print(f"调仓频率: {rebalance_freq}")
        print(f"训练窗口: {train_period} 天")
        
        # 获取交易日期
        trading_dates = self.data['date'].unique()
        trading_dates = pd.to_datetime(trading_dates)
        trading_dates = trading_dates[
            (trading_dates >= pd.to_datetime(start_date)) &
            (trading_dates <= pd.to_datetime(end_date))
        ]
        
        if len(trading_dates) == 0:
            raise ValueError(f"No trading dates in range {start_date} to {end_date}")
        
        # 确定调仓日期
        rebalance_dates = self._get_rebalance_dates(trading_dates, rebalance_freq)
        print(f"调仓次数: {len(rebalance_dates)}")
        
        # 重置组合
        self.portfolio.reset()
        
        # 逐期回测
        last_train_date = None
        
        for i, rebalance_date in enumerate(rebalance_dates):
            rebalance_date_str = rebalance_date.strftime('%Y-%m-%d')
            
            print(f"\n[{i+1}/{len(rebalance_dates)}] {rebalance_date_str}")
            
            # 判断是否需要重新训练
            if use_rolling_train:
                should_retrain = (
                    last_train_date is None or
                    (i % (retrain_freq // self._parse_freq_days(rebalance_freq)) == 0)
                )
                
                if should_retrain:
                    print("  重新训练模型...")
                    self._train_model(rebalance_date, train_period)
                    last_train_date = rebalance_date
            
            # 获取当期数据
            current_data = self.data[
                self.data['date'] == rebalance_date_str
            ].copy()
            
            if len(current_data) == 0:
                print(f"  警告: 无数据，跳过")
                continue
            
            # 选股
            try:
                X = current_data[self.feature_columns]
                stock_codes = current_data['stock_code'].tolist()
                
                selected_stocks = self.selector.select_stocks(
                    X, rebalance_date_str, stock_codes
                )
                
                print(f"  选中 {len(selected_stocks)} 只股票")
                
            except Exception as e:
                print(f"  选股失败: {e}")
                continue
            
            # 获取当前价格
            current_prices = dict(zip(
                current_data['stock_code'],
                current_data['close']
            ))
            
            # 调仓
            try:
                portfolio_status = self.portfolio.rebalance(
                    rebalance_date_str,
                    selected_stocks,
                    current_prices
                )
                
                self.results.append(portfolio_status)
                
            except Exception as e:
                print(f"  调仓失败: {e}")
                continue
            
            # 更新每日净值（从上次调仓到本次调仓之间的日期）
            if i > 0:
                prev_date = rebalance_dates[i-1]
                self._update_daily_values(prev_date, rebalance_date, current_prices)
        
        print("\n" + "="*80)
        print("回测完成！")
        print("="*80)
        
        results_df = pd.DataFrame(self.results)
        return results_df
    
    def _train_model(self, current_date: pd.Timestamp, train_period: int):
        """
        训练模型
        
        Args:
            current_date: 当前日期
            train_period: 训练期长度（天数）
        """
        # 获取训练数据
        end_date = current_date
        start_date = current_date - pd.Timedelta(days=train_period)
        
        train_data = self.data[
            (pd.to_datetime(self.data['date']) >= start_date) &
            (pd.to_datetime(self.data['date']) < end_date)
        ].copy()
        
        # 移除缺失标签
        train_data = train_data[train_data[self.label_col].notna()]
        
        if len(train_data) < 100:
            print(f"    警告: 训练数据不足 ({len(train_data)} 条)")
            return
        
        X_train = train_data[self.feature_columns]
        y_train = train_data[self.label_col]
        
        # 训练模型
        try:
            self.model.train(X_train, y_train, verbose=False)
            self.training_dates.append(current_date.strftime('%Y-%m-%d'))
            print(f"    训练完成: {len(train_data)} 条样本")
        except Exception as e:
            print(f"    训练失败: {e}")
    
    def _get_rebalance_dates(self,
                            trading_dates: pd.DatetimeIndex,
                            freq: str) -> List[pd.Timestamp]:
        """获取调仓日期"""
        if freq == '1D':
            return trading_dates.tolist()
        
        # 转换为DataFrame便于重采样
        df = pd.DataFrame({'date': trading_dates})
        df = df.set_index('date')
        
        # 重采样
        resampled = df.resample(freq).first()
        rebalance_dates = [idx for idx in resampled.index if idx in trading_dates]
        
        return rebalance_dates
    
    def _parse_freq_days(self, freq: str) -> int:
        """解析频率字符串为天数"""
        if freq == '1D':
            return 1
        elif freq == '5D':
            return 5
        elif freq == '20D':
            return 20
        elif freq == '60D':
            return 60
        else:
            return int(freq.replace('D', ''))
    
    def _update_daily_values(self,
                           start_date: pd.Timestamp,
                           end_date: pd.Timestamp,
                           prices: Dict[str, float]):
        """更新每日净值"""
        # 获取期间内所有交易日
        period_dates = self.data[
            (pd.to_datetime(self.data['date']) > start_date) &
            (pd.to_datetime(self.data['date']) <= end_date)
        ]['date'].unique()
        
        for date_str in period_dates:
            date_data = self.data[self.data['date'] == date_str]
            date_prices = dict(zip(date_data['stock_code'], date_data['close']))
            
            # 更新组合价值
            self.portfolio._update_portfolio_value(date_prices)
            self.portfolio._record_daily_position(date_str, date_prices)
    
    def get_results(self) -> pd.DataFrame:
        """获取回测结果"""
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame(self.results)
    
    def get_daily_returns(self) -> pd.Series:
        """获取每日收益率"""
        daily_values = self.portfolio.get_daily_values_df()
        
        if len(daily_values) == 0:
            return pd.Series()
        
        returns = daily_values['portfolio_value'].pct_change()
        returns.index = pd.to_datetime(daily_values['date'])
        
        return returns
    
    def get_equity_curve(self) -> pd.DataFrame:
        """获取净值曲线"""
        daily_values = self.portfolio.get_daily_values_df()
        
        if len(daily_values) == 0:
            return pd.DataFrame()
        
        equity_curve = daily_values[['date', 'portfolio_value']].copy()
        equity_curve['cumulative_return'] = (
            equity_curve['portfolio_value'] / self.portfolio.initial_capital - 1
        ) * 100
        
        return equity_curve
    
    def get_summary(self) -> Dict:
        """获取回测摘要"""
        daily_values = self.portfolio.get_daily_values_df()
        
        if len(daily_values) == 0:
            return {}
        
        final_value = daily_values['portfolio_value'].iloc[-1]
        initial_value = self.portfolio.initial_capital
        
        summary = {
            'start_date': daily_values['date'].iloc[0],
            'end_date': daily_values['date'].iloc[-1],
            'trading_days': len(daily_values),
            'initial_capital': initial_value,
            'final_value': final_value,
            'total_return': (final_value / initial_value - 1) * 100,
            'n_rebalances': len(self.results),
            'n_retrains': len(self.training_dates)
        }
        
        # 添加组合统计
        portfolio_stats = self.portfolio.get_statistics()
        summary.update(portfolio_stats)
        
        return summary


def test_backtester():
    """测试回测引擎"""
    print("="*60)
    print("测试回测引擎")
    print("="*60)
    
    # 创建模拟数据
    np.random.seed(42)
    n_dates = 100
    n_stocks = 50
    
    dates = pd.date_range('2024-01-01', periods=n_dates, freq='D')
    
    data = []
    for date in dates:
        for i in range(n_stocks):
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'stock_code': f'00{i:04d}',
                'close': np.random.rand() * 100 + 10,
                'factor1': np.random.randn(),
                'factor2': np.random.randn(),
                'factor3': np.random.randn(),
                'label': np.random.randint(0, 2)
            })
    
    df = pd.DataFrame(data)
    
    # 创建简单模型
    class SimpleModel:
        def train(self, X, y, verbose=False):
            pass
        
        def predict(self, X):
            return np.random.rand(len(X))
    
    # 创建组件
    from step6_stock_selector import StockSelector
    from step6_portfolio_manager import PortfolioManager
    
    model = SimpleModel()
    selector = StockSelector(model, n_stocks=10, method='top_n')
    portfolio = PortfolioManager(initial_capital=1000000)
    
    # 运行回测
    backtester = Backtester(
        model=model,
        stock_selector=selector,
        portfolio_manager=portfolio,
        data=df,
        feature_columns=['factor1', 'factor2', 'factor3']
    )
    
    results = backtester.run_backtest(
        start_date='2024-01-20',
        end_date='2024-04-09',
        rebalance_freq='10D',
        train_period=20,
        use_rolling_train=False
    )
    
    print("\n回测结果:")
    print(results)
    
    print("\n回测摘要:")
    summary = backtester.get_summary()
    for key, value in summary.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:,.2f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    test_backtester()
