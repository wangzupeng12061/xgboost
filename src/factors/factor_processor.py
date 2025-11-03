"""
因子预处理模块
包含去极值、标准化、中性化、缺失值处理等功能
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Optional, Tuple
from sklearn.linear_model import LinearRegression


class FactorProcessor:
    """因子预处理类"""
    
    def __init__(self, data: pd.DataFrame, factor_columns: List[str]):
        """
        初始化因子处理器
        
        Args:
            data: 包含因子的DataFrame
            factor_columns: 需要处理的因子列名列表
        """
        self.data = data.copy()
        self.factor_columns = factor_columns
        print(f"FactorProcessor initialized with {len(data)} rows and {len(factor_columns)} factors")
    
    def winsorize(self, 
                  method: str = 'mad',
                  n_std: float = 3,
                  quantile: Tuple[float, float] = (0.01, 0.99)) -> pd.DataFrame:
        """
        去极值处理
        
        Args:
            method: 方法 ('std' 标准差法, 'mad' 绝对中位差法, 'quantile' 分位数法)
            n_std: 标准差倍数
            quantile: 分位数上下限
            
        Returns:
            处理后的数据
        """
        print(f"\nWinsorizing factors using {method} method...")
        df = self.data.copy()
        
        for factor in self.factor_columns:
            if factor not in df.columns:
                print(f"Warning: {factor} not found in data")
                continue
            
            # 按日期分组处理（截面去极值）
            if method == 'std':
                # 3倍标准差法
                df[factor] = df.groupby('date')[factor].transform(
                    lambda x: self._winsorize_std(x, n_std)
                )
                
            elif method == 'mad':
                # MAD法（更稳健，推荐）
                df[factor] = df.groupby('date')[factor].transform(
                    lambda x: self._winsorize_mad(x, n_std)
                )
                
            elif method == 'quantile':
                # 分位数法
                df[factor] = df.groupby('date')[factor].transform(
                    lambda x: self._winsorize_quantile(x, quantile)
                )
        
        print("Winsorization completed")
        self.data = df
        return df
    
    @staticmethod
    def _winsorize_std(series: pd.Series, n_std: float) -> pd.Series:
        """标准差法去极值"""
        mean = series.mean()
        std = series.std()
        upper = mean + n_std * std
        lower = mean - n_std * std
        return series.clip(lower=lower, upper=upper)
    
    @staticmethod
    def _winsorize_mad(series: pd.Series, n_std: float) -> pd.Series:
        """MAD法去极值（中位数绝对偏差）"""
        median = series.median()
        mad = np.median(np.abs(series - median))
        
        # MAD与标准差的转换系数
        mad_std = 1.4826 * mad
        upper = median + n_std * mad_std
        lower = median - n_std * mad_std
        
        return series.clip(lower=lower, upper=upper)
    
    @staticmethod
    def _winsorize_quantile(series: pd.Series, quantile: Tuple[float, float]) -> pd.Series:
        """分位数法去极值"""
        lower_q = series.quantile(quantile[0])
        upper_q = series.quantile(quantile[1])
        return series.clip(lower=lower_q, upper=upper_q)
    
    def standardize(self, method: str = 'zscore') -> pd.DataFrame:
        """
        标准化处理
        
        Args:
            method: 方法 ('zscore', 'minmax', 'rank')
            
        Returns:
            标准化后的数据
        """
        print(f"\nStandardizing factors using {method} method...")
        df = self.data.copy()
        
        for factor in self.factor_columns:
            if factor not in df.columns:
                continue
            
            # 按日期分组标准化（截面标准化）
            if method == 'zscore':
                # Z-score标准化
                df[factor] = df.groupby('date')[factor].transform(
                    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
                )
                
            elif method == 'minmax':
                # Min-Max标准化到[0, 1]
                df[factor] = df.groupby('date')[factor].transform(
                    lambda x: (x - x.min()) / (x.max() - x.min()) 
                    if x.max() > x.min() else 0.5
                )
                
            elif method == 'rank':
                # 排序标准化（百分位排名）
                df[factor] = df.groupby('date')[factor].transform(
                    lambda x: x.rank(pct=True)
                )
        
        print("Standardization completed")
        self.data = df
        return df
    
    def neutralize(self,
                   industry_col: str = 'industry',
                   market_cap_col: str = 'market_cap') -> pd.DataFrame:
        """
        中性化处理（行业、市值中性化）
        
        Args:
            industry_col: 行业列名
            market_cap_col: 市值列名
            
        Returns:
            中性化后的数据
        """
        print(f"\nNeutralizing factors...")
        df = self.data.copy()
        
        if industry_col not in df.columns:
            print(f"Warning: {industry_col} not found, skipping neutralization")
            return df
        
        # 对每个因子进行中性化
        for factor in self.factor_columns:
            if factor not in df.columns:
                continue
            
            print(f"  Neutralizing {factor}...")
            
            # 按日期分组，对每个截面进行中性化
            df[factor] = df.groupby('date', group_keys=False).apply(
                lambda group: self._neutralize_cross_section(
                    group, factor, industry_col, market_cap_col
                )
            )[factor].values
        
        print("Neutralization completed")
        self.data = df
        return df
    
    @staticmethod
    def _neutralize_cross_section(group: pd.DataFrame,
                                  factor: str,
                                  industry_col: str,
                                  market_cap_col: str) -> pd.DataFrame:
        """对单个截面进行中性化"""
        if len(group) < 10:  # 样本太少，不进行中性化
            return group
        
        # 准备自变量
        X_list = []
        
        # 行业哑变量
        if industry_col in group.columns:
            industry_dummies = pd.get_dummies(
                group[industry_col], 
                prefix='industry',
                drop_first=True  # 避免多重共线性
            )
            X_list.append(industry_dummies)
        
        # 市值的对数
        if market_cap_col in group.columns:
            log_cap = np.log(group[market_cap_col].replace(0, np.nan))
            log_cap = log_cap.fillna(log_cap.median())
            X_list.append(pd.DataFrame({'log_market_cap': log_cap}, index=group.index))
        
        if not X_list:
            return group
        
        # 合并自变量
        X = pd.concat(X_list, axis=1)
        y = group[factor]
        
        # 处理缺失值
        valid_idx = y.notna() & X.notna().all(axis=1)
        
        if valid_idx.sum() < 10:
            return group
        
        try:
            # 线性回归
            model = LinearRegression()
            model.fit(X[valid_idx], y[valid_idx])
            
            # 预测值
            predictions = model.predict(X)
            
            # 残差作为中性化后的因子值
            group[factor] = y - predictions
            
        except Exception as e:
            print(f"    Warning: Neutralization failed for {factor}: {e}")
        
        return group
    
    def fill_missing(self, method: str = 'industry_median') -> pd.DataFrame:
        """
        缺失值填充
        
        Args:
            method: 填充方法 ('industry_median', 'cross_median', 'forward_fill', 'zero')
            
        Returns:
            填充后的数据
        """
        print(f"\nFilling missing values using {method} method...")
        df = self.data.copy()
        
        for factor in self.factor_columns:
            if factor not in df.columns:
                continue
            
            missing_count = df[factor].isna().sum()
            if missing_count == 0:
                continue
            
            print(f"  {factor}: {missing_count} missing values")
            
            if method == 'industry_median':
                # 用同行业同日期的中位数填充
                if 'industry' in df.columns:
                    df[factor] = df.groupby(['date', 'industry'])[factor].transform(
                        lambda x: x.fillna(x.median())
                    )
                    # 如果还有缺失，用全市场中位数填充
                    df[factor] = df.groupby('date')[factor].transform(
                        lambda x: x.fillna(x.median())
                    )
                else:
                    df[factor] = df.groupby('date')[factor].transform(
                        lambda x: x.fillna(x.median())
                    )
                    
            elif method == 'cross_median':
                # 用同日期全市场中位数填充
                df[factor] = df.groupby('date')[factor].transform(
                    lambda x: x.fillna(x.median())
                )
                
            elif method == 'forward_fill':
                # 用股票自身的历史值前向填充
                df[factor] = df.groupby('stock_code')[factor].fillna(method='ffill')
                
            elif method == 'zero':
                # 填充为0
                df[factor] = df[factor].fillna(0)
            
            remaining = df[factor].isna().sum()
            print(f"    Remaining: {remaining}")
        
        print("Missing value filling completed")
        self.data = df
        return df
    
    def process_pipeline(self,
                        winsorize_method: str = 'mad',
                        standardize_method: str = 'zscore',
                        neutralize: bool = True,
                        fill_method: str = 'industry_median') -> pd.DataFrame:
        """
        完整的因子预处理流程
        
        Args:
            winsorize_method: 去极值方法
            standardize_method: 标准化方法
            neutralize: 是否中性化
            fill_method: 缺失值填充方法
            
        Returns:
            处理后的数据
        """
        print("\n" + "="*60)
        print("Starting factor preprocessing pipeline...")
        print("="*60)
        
        # 1. 去极值
        self.winsorize(method=winsorize_method)
        
        # 2. 缺失值填充（在标准化之前）
        self.fill_missing(method=fill_method)
        
        # 3. 中性化（在标准化之前）
        if neutralize:
            self.neutralize()
        
        # 4. 标准化
        self.standardize(method=standardize_method)
        
        print("="*60)
        print("Factor preprocessing pipeline completed!")
        print("="*60 + "\n")
        
        return self.data
    
    def get_factor_stats(self) -> pd.DataFrame:
        """
        获取因子统计信息
        
        Returns:
            因子统计DataFrame
        """
        stats_list = []
        
        for factor in self.factor_columns:
            if factor not in self.data.columns:
                continue
            
            factor_data = self.data[factor]
            
            stats_list.append({
                'factor': factor,
                'mean': factor_data.mean(),
                'std': factor_data.std(),
                'min': factor_data.min(),
                'max': factor_data.max(),
                'missing_pct': factor_data.isna().sum() / len(factor_data) * 100,
                'unique_values': factor_data.nunique()
            })
        
        return pd.DataFrame(stats_list)


def test_factor_processor():
    """测试因子预处理功能"""
    # 创建测试数据
    np.random.seed(42)
    n_dates = 10
    n_stocks = 50
    
    dates = pd.date_range('2024-01-01', periods=n_dates, freq='D')
    
    test_data = []
    for date in dates:
        for i in range(n_stocks):
            test_data.append({
                'date': date,
                'stock_code': f'00000{i:02d}',
                'industry': f'Industry_{i % 5}',
                'market_cap': np.random.rand() * 1e10 + 1e9,
                'factor1': np.random.randn() * 100 + 10,
                'factor2': np.random.randn() * 50 + 5,
                'factor3': np.random.rand() * 0.5
            })
    
    df = pd.DataFrame(test_data)
    
    # 添加一些极值和缺失值
    df.loc[0, 'factor1'] = 1000  # 极值
    df.loc[5, 'factor2'] = np.nan  # 缺失值
    
    print("Original data stats:")
    print(df[['factor1', 'factor2', 'factor3']].describe())
    
    # 测试预处理
    processor = FactorProcessor(df, ['factor1', 'factor2', 'factor3'])
    
    # 完整流程
    processed = processor.process_pipeline(
        winsorize_method='mad',
        standardize_method='zscore',
        neutralize=True,
        fill_method='industry_median'
    )
    
    print("\nProcessed data stats:")
    print(processed[['factor1', 'factor2', 'factor3']].describe())
    
    # 查看因子统计
    print("\nFactor statistics:")
    print(processor.get_factor_stats())


if __name__ == "__main__":
    test_factor_processor()
