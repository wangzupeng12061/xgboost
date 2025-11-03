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
        
        # 如果已经有PE、PB、PS数据（来自市值数据），直接使用并计算倒数
        if 'pe' in df.columns:
            df['PE'] = df['pe']
            # EP (盈利收益率) = 1/PE
            df['EP'] = 1 / df['PE']
            df['EP'] = df['EP'].replace([np.inf, -np.inf], np.nan)
        
        if 'pb' in df.columns:
            df['PB'] = df['pb']
            # BP (账面市值比) = 1/PB
            df['BP'] = 1 / df['PB']
            df['BP'] = df['BP'].replace([np.inf, -np.inf], np.nan)
        
        if 'ps' in df.columns:
            df['PS'] = df['ps']
            # SP (销售收益率) = 1/PS
            df['SP'] = 1 / df['PS']
            df['SP'] = df['SP'].replace([np.inf, -np.inf], np.nan)
        
        # 市值因子
        if 'market_cap' in df.columns:
            # 对数市值（降低极端值影响）
            df['log_market_cap'] = np.log(df['market_cap'] + 1)
            
            # 市值分位数（按日期）
            df['market_cap_quantile'] = df.groupby('date')['market_cap'].rank(pct=True)
        
        if 'float_market_cap' in df.columns:
            df['log_float_cap'] = np.log(df['float_market_cap'] + 1)
        
        # 如果有完整的财务数据
        if 'eps' in df.columns and 'close' in df.columns:
            df['PE_calc'] = df['close'] / df['eps']
            df['PE_calc'] = df['PE_calc'].replace([np.inf, -np.inf], np.nan)
        
        if 'bvps' in df.columns and 'close' in df.columns:
            df['PB_calc'] = df['close'] / df['bvps']
            df['PB_calc'] = df['PB_calc'].replace([np.inf, -np.inf], np.nan)
        
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
        
        # 如果已有换手率数据（来自市值数据）
        if 'turnover_rate' in df.columns:
            # 平均换手率
            for window in [5, 20, 60]:
                df[f'avg_turnover_{window}d'] = (
                    df.groupby('stock_code')['turnover_rate']
                    .rolling(window).mean().reset_index(level=0, drop=True)
                )
            
            # 换手率标准差（流动性波动）
            df['turnover_std_20d'] = (
                df.groupby('stock_code')['turnover_rate']
                .rolling(20).std().reset_index(level=0, drop=True)
            )
        elif 'volume' in df.columns and 'float_shares' in df.columns:
            # 如果没有换手率，则计算
            df['turnover_rate'] = df['volume'] / df['float_shares']
            df['turnover_rate'] = df['turnover_rate'].replace([np.inf, -np.inf], np.nan)
            
            for window in [5, 20, 60]:
                df[f'avg_turnover_{window}d'] = (
                    df.groupby('stock_code')['turnover_rate']
                    .rolling(window).mean().reset_index(level=0, drop=True)
                )
        
        # 成交额（如果没有amount字段）
        if 'amount' not in df.columns and 'volume' in df.columns and 'close' in df.columns:
            df['amount'] = df['volume'] * df['close']
        
        # 成交额相关因子
        if 'amount' in df.columns:
            # 对数成交额（减少极端值影响）
            df['log_amount'] = np.log(df['amount'] + 1)
            
            # 平均成交额
            for window in [5, 20, 60]:
                df[f'avg_amount_{window}d'] = (
                    df.groupby('stock_code')['amount']
                    .rolling(window).mean().reset_index(level=0, drop=True)
                )
            
            # 成交额相对强度（与市场平均相比）
            avg_market_amount = df.groupby('date')['amount'].transform('mean')
            df['amount_relative'] = df['amount'] / avg_market_amount
            df['amount_relative'] = df['amount_relative'].replace([np.inf, -np.inf], np.nan)
        
        # Amihud非流动性指标
        if 'return' not in df.columns and 'close' in df.columns:
            df['return'] = df.groupby('stock_code')['close'].pct_change()
        
        if 'return' in df.columns and 'amount' in df.columns:
            df['amihud'] = np.abs(df['return']) / (df['amount'] + 1)
            df['amihud'] = df['amihud'].replace([np.inf, -np.inf], np.nan)
            
            # 平均Amihud
            df['amihud_20d'] = (
                df.groupby('stock_code')['amihud']
                .rolling(20).mean().reset_index(level=0, drop=True)
            )
        
        # 成交量相关因子
        if 'volume' in df.columns:
            # 对数成交量
            df['log_volume'] = np.log(df['volume'] + 1)
            
            # 成交量标准差
            df['volume_std_20d'] = (
                df.groupby('stock_code')['volume']
                .rolling(20).std().reset_index(level=0, drop=True)
            )
            
            # 成交量相对强度
            avg_market_volume = df.groupby('date')['volume'].transform('mean')
            df['volume_relative'] = df['volume'] / avg_market_volume
            df['volume_relative'] = df['volume_relative'].replace([np.inf, -np.inf], np.nan)
        
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
    
    # ==================== 价格形态因子 ====================
    def calculate_price_pattern_factors(self) -> pd.DataFrame:
        """
        计算价格形态因子
        
        Returns:
            包含价格形态因子的DataFrame
        """
        print("Calculating price pattern factors...")
        df = self.data.copy()
        
        if 'close' not in df.columns:
            print("Warning: 'close' column not found")
            return df
        
        # 价格位置因子
        if all(col in df.columns for col in ['high', 'low', 'close']):
            # 当前价格在日内高低价的位置
            df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            df['price_position'] = df['price_position'].replace([np.inf, -np.inf], np.nan)
            
            # 价格相对于N日高点的位置
            for window in [20, 60]:
                df[f'high_{window}d'] = df.groupby('stock_code')['high'].transform(
                    lambda x: x.rolling(window).max()
                )
                df[f'low_{window}d'] = df.groupby('stock_code')['low'].transform(
                    lambda x: x.rolling(window).min()
                )
                df[f'price_position_{window}d'] = (
                    (df['close'] - df[f'low_{window}d']) / 
                    (df[f'high_{window}d'] - df[f'low_{window}d'])
                )
                df[f'price_position_{window}d'] = df[f'price_position_{window}d'].replace([np.inf, -np.inf], np.nan)
        
        # 价格动量形态
        if 'open' in df.columns and 'close' in df.columns:
            # 实体大小（实体/价格）
            df['candle_body'] = np.abs(df['close'] - df['open']) / df['close']
            df['candle_body'] = df['candle_body'].replace([np.inf, -np.inf], np.nan)
            
            # 上影线/下影线比例
            if all(col in df.columns for col in ['high', 'low']):
                df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
                df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
                df['shadow_ratio'] = df['upper_shadow'] / (df['lower_shadow'] + 1e-6)
                df['shadow_ratio'] = df['shadow_ratio'].replace([np.inf, -np.inf], np.nan)
        
        # 均线系统
        for window in [5, 10, 20, 60]:
            df[f'MA{window}'] = df.groupby('stock_code')['close'].transform(
                lambda x: x.rolling(window).mean()
            )
            # 价格相对均线的偏离度
            df[f'close_to_ma{window}'] = (df['close'] - df[f'MA{window}']) / df[f'MA{window}']
            df[f'close_to_ma{window}'] = df[f'close_to_ma{window}'].replace([np.inf, -np.inf], np.nan)
        
        # 均线多头/空头排列
        if all(f'MA{w}' in df.columns for w in [5, 10, 20]):
            df['ma_trend'] = (
                (df['MA5'] > df['MA10']).astype(int) +
                (df['MA10'] > df['MA20']).astype(int)
            ) / 2  # 归一化到[0, 1]
        
        # 价格突破
        if 'close' in df.columns:
            # 创N日新高/新低
            for window in [20, 60]:
                high_rolling = df.groupby('stock_code')['close'].transform(
                    lambda x: x.rolling(window).max()
                )
                low_rolling = df.groupby('stock_code')['close'].transform(
                    lambda x: x.rolling(window).min()
                )
                df[f'new_high_{window}d'] = (df['close'] >= high_rolling).astype(int)
                df[f'new_low_{window}d'] = (df['close'] <= low_rolling).astype(int)
        
        print("Price pattern factors calculated")
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
        
        # 检查数据中是否有财务数据或市值数据
        has_financial_data = any(col in df.columns for col in ['eps', 'bvps', 'revenue', 'net_profit'])
        has_valuation_data = any(col in df.columns for col in ['pe', 'pb', 'ps', 'market_cap'])
        has_financial_indicators = any(col in df.columns for col in ['roe', 'roa', 'debt_to_assets'])
        has_industry_data = 'industry' in df.columns
        has_macro_data = any(col in df.columns for col in ['cpi_yoy', 'pmi', 'm2_yoy', 'shibor_on', 'gdp_yoy'])
        
        # 估值因子（如果有PE/PB/PS或市值数据）
        if has_financial_data or has_valuation_data:
            df = self.calculate_valuation_factors()
            self.data = df
            print(f"  ✓ Valuation factors completed")
        else:
            print("  - Skipping valuation factors (no financial/market data)")
        
        # 财务成长和盈利因子（需要完整财务数据）
        if has_financial_data:
            df = self.calculate_growth_factors()
            self.data = df
            print(f"  ✓ Growth factors completed")
            
            df = self.calculate_profitability_factors()
            self.data = df
            print(f"  ✓ Profitability factors completed")
        else:
            print("  - Skipping growth/profitability factors (no financial data)")
        
        # 质量因子（基于财务指标）
        if has_financial_indicators:
            df = self.calculate_quality_factors()
            self.data = df
            print(f"  ✓ Quality factors completed")
        else:
            print("  - Skipping quality factors (no financial indicators)")
        
        # 基于价格数据的因子（总是计算）
        df = self.calculate_momentum_factors()
        self.data = df
        print(f"  ✓ Momentum factors completed")
        
        df = self.calculate_volatility_factors()
        self.data = df
        print(f"  ✓ Volatility factors completed")
        
        df = self.calculate_liquidity_factors()
        self.data = df
        print(f"  ✓ Liquidity factors completed")
        
        df = self.calculate_technical_factors()
        self.data = df
        print(f"  ✓ Technical factors completed")
        
        df = self.calculate_price_pattern_factors()
        self.data = df
        print(f"  ✓ Price pattern factors completed")
        
        # 行业轮动因子（如果有行业数据）
        if has_industry_data:
            df = self.calculate_industry_factors()
            self.data = df
            print(f"  ✓ Industry rotation factors completed")
        else:
            print("  - Skipping industry factors (no industry data)")
        
        # 宏观因子（如果有宏观数据）
        if has_macro_data:
            df = self.calculate_macro_factors()
            self.data = df
            print(f"  ✓ Macro sensitivity factors completed")
        else:
            print("  - Skipping macro factors (no macro data)")
        
        print("="*60)
        print("Factor calculation completed!")
        print("="*60 + "\n")
        
        self.factors = df
        return df
    
    # ==================== 行业轮动因子 ====================
    def calculate_industry_factors(self) -> pd.DataFrame:
        """
        计算行业轮动因子
        需要数据包含industry列
        
        Returns:
            包含行业因子的DataFrame
        """
        print("Calculating industry rotation factors...")
        df = self.data.copy()
        
        if 'industry' not in df.columns:
            print("  Warning: 'industry' column not found, skipping industry factors")
            return df
        
        # 1. 行业相对强度 (Industry Relative Strength)
        # 计算个股收益与行业平均收益的差
        for period in [5, 20, 60]:
            # 个股收益
            df[f'stock_return_{period}d'] = df.groupby('stock_code')['close'].pct_change(period)
            
            # 行业平均收益
            industry_return = df.groupby(['date', 'industry'])[f'stock_return_{period}d'].transform('mean')
            df[f'industry_return_{period}d'] = industry_return
            
            # 相对强度 = 个股收益 - 行业收益
            df[f'industry_relative_strength_{period}d'] = df[f'stock_return_{period}d'] - df[f'industry_return_{period}d']
            
            # 清理临时列
            df = df.drop([f'stock_return_{period}d', f'industry_return_{period}d'], axis=1)
        
        # 2. 行业动量 (Industry Momentum)
        # 个股收益在行业内的排名百分位
        for period in [20, 60]:
            df[f'temp_return_{period}d'] = df.groupby('stock_code')['close'].pct_change(period)
            df[f'industry_momentum_rank_{period}d'] = df.groupby(['date', 'industry'])[f'temp_return_{period}d'].rank(pct=True)
            df = df.drop(f'temp_return_{period}d', axis=1)
        
        # 3. 行业集中度 (Industry Concentration)
        # 股票市值在行业内的占比
        if 'market_cap' in df.columns:
            industry_total_cap = df.groupby(['date', 'industry'])['market_cap'].transform('sum')
            df['industry_concentration'] = df['market_cap'] / industry_total_cap
            df['industry_concentration'] = df['industry_concentration'].fillna(0)
        
        print("  ✓ Industry factors completed")
        return df
    
    # ==================== 质量因子 (基于财务指标) ====================
    def calculate_quality_factors(self) -> pd.DataFrame:
        """
        计算质量因子
        需要数据包含财务指标列
        
        Returns:
            包含质量因子的DataFrame
        """
        print("Calculating quality factors...")
        df = self.data.copy()
        
        # 检查是否有财务指标数据
        financial_cols = ['roe', 'roa', 'debt_to_assets', 'gross_profit_margin', 'net_profit_margin']
        available_cols = [col for col in financial_cols if col in df.columns]
        
        if len(available_cols) == 0:
            print("  Warning: No financial indicators found, skipping quality factors")
            return df
        
        # 1. 盈利能力 (Profitability)
        if 'roe' in df.columns:
            df['ROE'] = df['roe']
        
        if 'roa' in df.columns:
            df['ROA'] = df['roa']
        
        if 'gross_profit_margin' in df.columns:
            df['gross_margin'] = df['gross_profit_margin']
        
        if 'net_profit_margin' in df.columns:
            df['net_margin'] = df['net_profit_margin']
        
        # 2. 财务健康度 (Financial Health)
        if 'debt_to_assets' in df.columns:
            df['debt_ratio'] = df['debt_to_assets']
            # 低负债率=高健康度
            df['financial_health'] = 1 - df['debt_ratio']
        
        # 3. 成长性 (Growth)
        if 'revenue_yoy' in df.columns:
            df['revenue_growth'] = df['revenue_yoy'] / 100  # 转为小数
        
        if 'net_profit_yoy' in df.columns:
            df['profit_growth'] = df['net_profit_yoy'] / 100
        
        # 4. 综合质量得分 (Composite Quality Score)
        # 简单平均可用的质量指标（标准化后）
        quality_components = []
        if 'roe' in df.columns:
            quality_components.append(df['roe'].rank(pct=True))
        if 'roa' in df.columns:
            quality_components.append(df['roa'].rank(pct=True))
        if 'debt_to_assets' in df.columns:
            quality_components.append(1 - df['debt_to_assets'].rank(pct=True))  # 低负债好
        
        if len(quality_components) > 0:
            df['quality_score'] = pd.concat(quality_components, axis=1).mean(axis=1)
        
        print(f"  ✓ Quality factors completed ({len(available_cols)} indicators used)")
        return df
    
    # ==================== 宏观因子 ====================
    def calculate_macro_factors(self) -> pd.DataFrame:
        """
        计算宏观敏感度因子
        需要数据包含宏观经济指标列
        
        Returns:
            包含宏观因子的DataFrame
        """
        print("Calculating macro sensitivity factors...")
        df = self.data.copy()
        
        # 检查是否有宏观数据
        macro_cols = ['cpi_yoy', 'pmi', 'm2_yoy', 'shibor_on', 'gdp_yoy']
        available_macro = [col for col in macro_cols if col in df.columns]
        
        if len(available_macro) == 0:
            print("  Warning: No macro data found, skipping macro factors")
            return df
        
        print(f"  Available macro indicators: {available_macro}")
        
        # 确保有收益率数据
        if 'return' not in df.columns:
            df['return'] = df.groupby('stock_code')['close'].pct_change()
        
        # 1. 利率敏感度 (Interest Rate Beta)
        if 'shibor_on' in df.columns:
            # 计算利率变化
            df['shibor_change'] = df.groupby('stock_code')['shibor_on'].diff()
            
            # 滚动窗口计算股票收益与利率的相关性
            def rolling_correlation(group, window=60):
                if len(group) < window:
                    return pd.Series(index=group.index, dtype=float)
                return group['return'].rolling(window).corr(group['shibor_change'])
            
            df['interest_rate_beta'] = df.groupby('stock_code', group_keys=False).apply(
                lambda x: x['return'].rolling(60, min_periods=30).corr(x['shibor_change'])
            ).reset_index(level=0, drop=True)
            
            # 利率敏感度的波动性
            df['interest_rate_volatility'] = df.groupby('stock_code')['interest_rate_beta'].transform(
                lambda x: x.rolling(20, min_periods=10).std()
            )
        
        # 2. 通胀敏感度 (Inflation Beta)
        if 'cpi_yoy' in df.columns:
            # CPI变化率
            df['cpi_change'] = df.groupby('stock_code')['cpi_yoy'].diff()
            
            # 股票收益与CPI的滚动相关性
            df['inflation_beta'] = df.groupby('stock_code', group_keys=False).apply(
                lambda x: x['return'].rolling(60, min_periods=30).corr(x['cpi_change'])
            ).reset_index(level=0, drop=True)
            
            # 高通胀Beta表示该股票在通胀上升时表现好
        
        # 3. 经济周期敏感度 (Economic Cycle Beta)
        if 'pmi' in df.columns:
            # PMI变化
            df['pmi_change'] = df.groupby('stock_code')['pmi'].diff()
            
            # 股票收益与PMI的相关性
            df['cycle_beta'] = df.groupby('stock_code', group_keys=False).apply(
                lambda x: x['return'].rolling(60, min_periods=30).corr(x['pmi_change'])
            ).reset_index(level=0, drop=True)
            
            # PMI > 50表示经济扩张，< 50表示收缩
            df['pmi_regime'] = (df['pmi'] > 50).astype(int)
        
        # 4. 货币政策敏感度 (Monetary Policy Beta)
        if 'm2_yoy' in df.columns:
            # M2增速变化
            df['m2_change'] = df.groupby('stock_code')['m2_yoy'].diff()
            
            # 股票收益与M2的相关性
            df['monetary_beta'] = df.groupby('stock_code', group_keys=False).apply(
                lambda x: x['return'].rolling(60, min_periods=30).corr(x['m2_change'])
            ).reset_index(level=0, drop=True)
        
        # 5. GDP增长敏感度 (GDP Growth Beta)
        if 'gdp_yoy' in df.columns:
            # GDP增速变化
            df['gdp_change'] = df.groupby('stock_code')['gdp_yoy'].diff()
            
            # 股票收益与GDP的相关性
            df['gdp_beta'] = df.groupby('stock_code', group_keys=False).apply(
                lambda x: x['return'].rolling(120, min_periods=60).corr(x['gdp_change'])
            ).reset_index(level=0, drop=True)
        
        # 6. 宏观风险综合得分 (Macro Risk Score)
        # 计算股票对宏观因素的总体敏感度
        macro_beta_cols = [col for col in df.columns if col.endswith('_beta')]
        if len(macro_beta_cols) > 0:
            # 使用绝对值，高敏感度 = 高风险
            df['macro_risk_score'] = df[macro_beta_cols].abs().mean(axis=1)
        
        # 7. 宏观环境指数 (Macro Environment Index)
        # 综合当前宏观环境的强弱
        macro_components = []
        if 'pmi' in df.columns:
            # PMI标准化（50为中性）
            macro_components.append((df['pmi'] - 50) / 10)
        if 'cpi_yoy' in df.columns:
            # CPI标准化（2%为中性目标）
            macro_components.append((2 - df['cpi_yoy']) / 2)  # 低通胀好
        if 'm2_yoy' in df.columns:
            # M2增速标准化（10%为中性）
            macro_components.append((df['m2_yoy'] - 10) / 5)
        
        if len(macro_components) > 0:
            df['macro_environment_index'] = pd.concat(macro_components, axis=1).mean(axis=1)
        
        # 清理临时列
        temp_cols = ['shibor_change', 'cpi_change', 'pmi_change', 'm2_change', 'gdp_change']
        df = df.drop([col for col in temp_cols if col in df.columns], axis=1)
        
        print(f"  ✓ Macro factors completed ({len(available_macro)} macro indicators used)")
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
