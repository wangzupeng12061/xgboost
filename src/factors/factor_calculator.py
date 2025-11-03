"""
因子计算模块
计算各类因子：估值、成长、盈利、质量、动量、波动、流动性、技术指标
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class FactorCalculator:
    """因子计算类 - 完整版本"""
    
    def __init__(self, data: pd.DataFrame):
        """
        初始化因子计算器
        
        Args:
            data: 包含价格和财务数据的DataFrame
                 必须包含: stock_code, date, close等列
        """
        self.data = data.sort_values(['stock_code', 'date']).reset_index(drop=True)
        self.factors = pd.DataFrame()
        print(f"FactorCalculator initialized with {len(data)} rows")
    
    # ==================== 估值因子 ====================
    def calculate_valuation_factors(self) -> pd.DataFrame:
        """
        计算估值因子
        
        Returns:
            包含估值因子的DataFrame
        """
        print("Calculating valuation factors...")
        df = self.data.copy()
        
        # PE (市盈率)
        if 'eps' in df.columns and 'close' in df.columns:
            df['PE'] = df['close'] / df['eps']
            df['PE'] = df['PE'].replace([np.inf, -np.inf], np.nan)
            # EP (盈利收益率) = 1/PE
            df['EP'] = 1 / df['PE']
            df['EP'] = df['EP'].replace([np.inf, -np.inf], np.nan)
        
        # PB (市净率)
        if 'bvps' in df.columns and 'close' in df.columns:
            df['PB'] = df['close'] / df['bvps']
            df['PB'] = df['PB'].replace([np.inf, -np.inf], np.nan)
            # BP (账面市值比) = 1/PB
            df['BP'] = 1 / df['PB']
            df['BP'] = df['BP'].replace([np.inf, -np.inf], np.nan)
        
        # PS (市销率)
        if 'revenue' in df.columns and 'market_cap' in df.columns:
            # 每股营收
            df['revenue_per_share'] = df['revenue'] / df['total_shares']
            df['PS'] = df['close'] / df['revenue_per_share']
            df['PS'] = df['PS'].replace([np.inf, -np.inf], np.nan)
        
        # EV/EBITDA
        if all(col in df.columns for col in ['market_cap', 'total_debt', 'cash', 'ebitda']):
            df['EV'] = df['market_cap'] + df['total_debt'] - df['cash']
            df['EV_EBITDA'] = df['EV'] / df['ebitda']
            df['EV_EBITDA'] = df['EV_EBITDA'].replace([np.inf, -np.inf], np.nan)
        
        print(f"Valuation factors calculated")
        return df
    
    # ==================== 成长因子 ====================
    def calculate_growth_factors(self) -> pd.DataFrame:
        """
        计算成长因子
        
        Returns:
            包含成长因子的DataFrame
        """
        print("Calculating growth factors...")
        df = self.data.copy()
        
        # 营收增长率 (YoY - 同比)
        if 'revenue' in df.columns:
            # 假设季度数据，4期前为去年同期
            df['revenue_growth'] = df.groupby('stock_code')['revenue'].pct_change(4)
            
            # 3年复合增长率 (CAGR)
            df['revenue_cagr_3y'] = (
                (df.groupby('stock_code')['revenue'].shift(0) / 
                 df.groupby('stock_code')['revenue'].shift(12)) ** (1/3) - 1
            )
        
        # 净利润增长率
        if 'net_profit' in df.columns:
            df['profit_growth'] = df.groupby('stock_code')['net_profit'].pct_change(4)
        
        # ROE增长率
        if 'roe' in df.columns:
            df['roe_growth'] = df.groupby('stock_code')['roe'].diff(4)
        
        # 营业利润增长率
        if 'operating_profit' in df.columns:
            df['op_growth'] = df.groupby('stock_code')['operating_profit'].pct_change(4)
        
        print("Growth factors calculated")
        return df
    
    # ==================== 盈利因子 ====================
    def calculate_profitability_factors(self) -> pd.DataFrame:
        """
        计算盈利因子
        
        Returns:
            包含盈利因子的DataFrame
        """
        print("Calculating profitability factors...")
        df = self.data.copy()
        
        # ROE (净资产收益率)
        if 'net_profit' in df.columns and 'total_equity' in df.columns:
            df['ROE'] = df['net_profit'] / df['total_equity']
            df['ROE'] = df['ROE'].replace([np.inf, -np.inf], np.nan)
        
        # ROA (总资产收益率)
        if 'net_profit' in df.columns and 'total_assets' in df.columns:
            df['ROA'] = df['net_profit'] / df['total_assets']
            df['ROA'] = df['ROA'].replace([np.inf, -np.inf], np.nan)
        
        # 毛利率
        if 'revenue' in df.columns and 'cost' in df.columns:
            df['gross_margin'] = (df['revenue'] - df['cost']) / df['revenue']
            df['gross_margin'] = df['gross_margin'].replace([np.inf, -np.inf], np.nan)
        
        # 净利率
        if 'net_profit' in df.columns and 'revenue' in df.columns:
            df['net_margin'] = df['net_profit'] / df['revenue']
            df['net_margin'] = df['net_margin'].replace([np.inf, -np.inf], np.nan)
        
        # ROIC (投资资本回报率)
        if all(col in df.columns for col in ['nopat', 'total_equity', 'total_debt']):
            df['invested_capital'] = df['total_equity'] + df['total_debt']
            df['ROIC'] = df['nopat'] / df['invested_capital']
            df['ROIC'] = df['ROIC'].replace([np.inf, -np.inf], np.nan)
        
        print("Profitability factors calculated")
        return df
    
    # ==================== 质量因子 ====================
    def calculate_quality_factors(self) -> pd.DataFrame:
        """
        计算质量因子
        
        Returns:
            包含质量因子的DataFrame
        """
        print("Calculating quality factors...")
        df = self.data.copy()
        
        # 资产负债率
        if 'total_debt' in df.columns and 'total_assets' in df.columns:
            df['debt_to_asset'] = df['total_debt'] / df['total_assets']
            df['debt_to_asset'] = df['debt_to_asset'].replace([np.inf, -np.inf], np.nan)
        
        # 流动比率
        if 'current_assets' in df.columns and 'current_liabilities' in df.columns:
            df['current_ratio'] = df['current_assets'] / df['current_liabilities']
            df['current_ratio'] = df['current_ratio'].replace([np.inf, -np.inf], np.nan)
        
        # 速动比率
        if all(col in df.columns for col in ['current_assets', 'inventory', 'current_liabilities']):
            df['quick_ratio'] = (df['current_assets'] - df['inventory']) / df['current_liabilities']
            df['quick_ratio'] = df['quick_ratio'].replace([np.inf, -np.inf], np.nan)
        
        # 经营现金流/净利润
        if 'operating_cash_flow' in df.columns and 'net_profit' in df.columns:
            df['ocf_to_profit'] = df['operating_cash_flow'] / df['net_profit']
            df['ocf_to_profit'] = df['ocf_to_profit'].replace([np.inf, -np.inf], np.nan)
        
        # 应收账款周转率
        if 'revenue' in df.columns and 'accounts_receivable' in df.columns:
            df['receivable_turnover'] = df['revenue'] / df['accounts_receivable']
            df['receivable_turnover'] = df['receivable_turnover'].replace([np.inf, -np.inf], np.nan)
        
        # 资产周转率
        if 'revenue' in df.columns and 'total_assets' in df.columns:
            df['asset_turnover'] = df['revenue'] / df['total_assets']
            df['asset_turnover'] = df['asset_turnover'].replace([np.inf, -np.inf], np.nan)
        
        print("Quality factors calculated")
        return df
    
    # ==================== 动量因子 ====================
    def calculate_momentum_factors(self, windows: List[int] = [5, 20, 60, 120]) -> pd.DataFrame:
        """
        计算动量因子
        
        Args:
            windows: 动量计算窗口列表
            
        Returns:
            包含动量因子的DataFrame
        """
        print(f"Calculating momentum factors with windows: {windows}...")
        df = self.data.copy()
        
        if 'close' not in df.columns:
            print("Warning: 'close' column not found, skipping momentum factors")
            return df
        
        for window in windows:
            # 收益率
            df[f'return_{window}d'] = (
                df.groupby('stock_code')['close'].pct_change(window)
            )
            
            # 如果有市场收益率，计算超额收益
            if 'market_return' in df.columns:
                df[f'excess_return_{window}d'] = (
                    df[f'return_{window}d'] - df['market_return']
                )
        
        # RSI (相对强弱指标)
        df['RSI_14'] = df.groupby('stock_code').apply(
            lambda x: self._calculate_rsi(x, period=14)
        ).reset_index(level=0, drop=True)
        
        print("Momentum factors calculated")
        return df
    
    @staticmethod
    def _calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        计算RSI指标
        
        Args:
            df: 数据框
            period: 周期
            
        Returns:
            RSI序列
        """
        if 'close' not in df.columns or len(df) < period:
            return pd.Series([np.nan] * len(df), index=df.index)
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    # ==================== 波动率因子 ====================
    def calculate_volatility_factors(self, windows: List[int] = [20, 60]) -> pd.DataFrame:
        """
        计算波动率因子
        
        Args:
            windows: 波动率计算窗口列表
            
        Returns:
            包含波动率因子的DataFrame
        """
        print(f"Calculating volatility factors with windows: {windows}...")
        df = self.data.copy()
        
        if 'close' not in df.columns:
            print("Warning: 'close' column not found, skipping volatility factors")
            return df
        
        # 计算收益率
        df['return'] = df.groupby('stock_code')['close'].pct_change()
        
        for window in windows:
            # 历史波动率 (年化)
            df[f'volatility_{window}d'] = (
                df.groupby('stock_code')['return']
                .rolling(window).std().reset_index(level=0, drop=True) * np.sqrt(252)
            )
            
            # 最大回撤
            df[f'max_drawdown_{window}d'] = (
                df.groupby('stock_code')['close']
                .rolling(window)
                .apply(lambda x: self._calculate_max_drawdown(x), raw=False)
                .reset_index(level=0, drop=True)
            )
        
        print("Volatility factors calculated")
        return df
    
    @staticmethod
    def _calculate_max_drawdown(prices: pd.Series) -> float:
        """
        计算最大回撤
        
        Args:
            prices: 价格序列
            
        Returns:
            最大回撤值
        """
        if len(prices) < 2:
            return 0
        
        cummax = prices.expanding().max()
        drawdown = (prices - cummax) / cummax
        return drawdown.min()
    
    # ==================== 流动性因子 ====================
    def calculate_liquidity_factors(self) -> pd.DataFrame:
        """
        计算流动性因子
        
        Returns:
            包含流动性因子的DataFrame
        """
        print("Calculating liquidity factors...")
        df = self.data.copy()
        
        # 换手率
        if 'volume' in df.columns and 'float_shares' in df.columns:
            df['turnover_rate'] = df['volume'] / df['float_shares']
            df['turnover_rate'] = df['turnover_rate'].replace([np.inf, -np.inf], np.nan)
            
            # 平均换手率
            for window in [5, 20, 60]:
                df[f'avg_turnover_{window}d'] = (
                    df.groupby('stock_code')['turnover_rate']
                    .rolling(window).mean().reset_index(level=0, drop=True)
                )
        
        # 成交额
        if 'volume' in df.columns and 'close' in df.columns:
            df['amount'] = df['volume'] * df['close']
            
            # 平均成交额
            for window in [5, 20, 60]:
                df[f'avg_amount_{window}d'] = (
                    df.groupby('stock_code')['amount']
                    .rolling(window).mean().reset_index(level=0, drop=True)
                )
        
        # Amihud非流动性指标
        if 'return' in df.columns and 'amount' in df.columns:
            df['amihud'] = np.abs(df['return']) / df['amount']
            df['amihud'] = df['amihud'].replace([np.inf, -np.inf], np.nan)
        
        # 成交量标准差
        if 'volume' in df.columns:
            df['volume_std_20d'] = (
                df.groupby('stock_code')['volume']
                .rolling(20).std().reset_index(level=0, drop=True)
            )
        
        print("Liquidity factors calculated")
        return df
    
    # ==================== 技术指标因子 ====================
    def calculate_technical_factors(self) -> pd.DataFrame:
        """
        计算技术指标因子
        
        Returns:
            包含技术指标的DataFrame
        """
        print("Calculating technical factors...")
        df = self.data.copy()
        
        if 'close' not in df.columns:
            print("Warning: 'close' column not found")
            return df
        
        # MACD
        df = self._calculate_macd(df)
        
        # 布林带
        df = self._calculate_bollinger_bands(df)
        
        # KDJ
        if all(col in df.columns for col in ['high', 'low', 'close']):
            df = self._calculate_kdj(df)
        
        print("Technical factors calculated")
        return df
    
    def _calculate_macd(self, df: pd.DataFrame, 
                       fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        计算MACD指标
        
        Args:
            df: 数据框
            fast: 快线周期
            slow: 慢线周期
            signal: 信号线周期
            
        Returns:
            添加MACD指标的数据框
        """
        # EMA快线
        df['EMA12'] = df.groupby('stock_code')['close'].transform(
            lambda x: x.ewm(span=fast, adjust=False).mean()
        )
        
        # EMA慢线
        df['EMA26'] = df.groupby('stock_code')['close'].transform(
            lambda x: x.ewm(span=slow, adjust=False).mean()
        )
        
        # MACD线 = 快线 - 慢线
        df['MACD'] = df['EMA12'] - df['EMA26']
        
        # 信号线
        df['Signal'] = df.groupby('stock_code')['MACD'].transform(
            lambda x: x.ewm(span=signal, adjust=False).mean()
        )
        
        # MACD柱
        df['MACD_hist'] = df['MACD'] - df['Signal']
        
        return df
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, 
                                   period: int = 20, std_dev: int = 2) -> pd.DataFrame:
        """
        计算布林带指标
        
        Args:
            df: 数据框
            period: 周期
            std_dev: 标准差倍数
            
        Returns:
            添加布林带指标的数据框
        """
        # 中轨 (移动平均)
        df['BB_middle'] = df.groupby('stock_code')['close'].transform(
            lambda x: x.rolling(period).mean()
        )
        
        # 标准差
        df['BB_std'] = df.groupby('stock_code')['close'].transform(
            lambda x: x.rolling(period).std()
        )
        
        # 上轨
        df['BB_upper'] = df['BB_middle'] + std_dev * df['BB_std']
        
        # 下轨
        df['BB_lower'] = df['BB_middle'] - std_dev * df['BB_std']
        
        # 布林带位置 (0-1之间)
        df['BB_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        df['BB_position'] = df['BB_position'].clip(0, 1)
        
        # 布林带宽度
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        
        return df
    
    def _calculate_kdj(self, df: pd.DataFrame, 
                      period: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
        """
        计算KDJ指标
        
        Args:
            df: 数据框
            period: RSV周期
            m1: K值平滑参数
            m2: D值平滑参数
            
        Returns:
            添加KDJ指标的数据框
        """
        # 计算RSV
        df['lowest_low'] = df.groupby('stock_code')['low'].transform(
            lambda x: x.rolling(period).min()
        )
        df['highest_high'] = df.groupby('stock_code')['high'].transform(
            lambda x: x.rolling(period).max()
        )
        
        df['RSV'] = 100 * (df['close'] - df['lowest_low']) / (df['highest_high'] - df['lowest_low'])
        df['RSV'] = df['RSV'].fillna(50)  # 填充初始值
        
        # 计算K值
        df['K'] = df.groupby('stock_code')['RSV'].transform(
            lambda x: x.ewm(alpha=1/m1, adjust=False).mean()
        )
        
        # 计算D值
        df['D'] = df.groupby('stock_code')['K'].transform(
            lambda x: x.ewm(alpha=1/m2, adjust=False).mean()
        )
        
        # 计算J值
        df['J'] = 3 * df['K'] - 2 * df['D']
        
        return df
    
    # ==================== 计算所有因子 ====================
    def calculate_all_factors(self) -> pd.DataFrame:
        """
        计算所有因子
        
        Returns:
            包含所有因子的DataFrame
        """
        print("\n" + "="*60)
        print("Starting factor calculation...")
        print("="*60)
        
        df = self.data.copy()
        
        # 依次计算各类因子
        df = self.calculate_valuation_factors()
        self.data = df
        
        df = self.calculate_growth_factors()
        self.data = df
        
        df = self.calculate_profitability_factors()
        self.data = df
        
        df = self.calculate_quality_factors()
        self.data = df
        
        df = self.calculate_momentum_factors()
        self.data = df
        
        df = self.calculate_volatility_factors()
        self.data = df
        
        df = self.calculate_liquidity_factors()
        self.data = df
        
        df = self.calculate_technical_factors()
        self.data = df
        
        print("="*60)
        print("Factor calculation completed!")
        print("="*60 + "\n")
        
        self.factors = df
        return df


def test_factor_calculator():
    """测试因子计算功能"""
    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    test_data = pd.DataFrame({
        'date': dates.tolist() * 2,
        'stock_code': ['000001']*100 + ['000002']*100,
        'close': np.random.rand(200) * 50 + 10,
        'open': np.random.rand(200) * 50 + 10,
        'high': np.random.rand(200) * 50 + 15,
        'low': np.random.rand(200) * 50 + 5,
        'volume': np.random.rand(200) * 1000000,
        'float_shares': [100000000] * 200
    })
    
    # 添加一些必要字段
    test_data['amount'] = test_data['volume'] * test_data['close']
    test_data['return'] = test_data.groupby('stock_code')['close'].pct_change()
    
    print("Testing FactorCalculator...")
    calc = FactorCalculator(test_data)
    
    # 测试所有因子
    result = calc.calculate_all_factors()
    
    print(f"\nTotal columns: {len(result.columns)}")
    print(f"\nFactor columns: {[col for col in result.columns if col not in test_data.columns][:20]}")
    
    print("\nSample data with factors:")
    print(result[['date', 'stock_code', 'close', 'MACD_hist', 'BB_position']].tail(10))


if __name__ == "__main__":
    test_factor_calculator()
