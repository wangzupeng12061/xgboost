"""
数据处理模块
包含数据清洗、缺失值处理、数据合并等功能
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional


class DataProcessor:
    """数据处理类"""
    
    def __init__(self):
        """初始化数据处理器"""
        pass
    
    @staticmethod
    def clean_data(df: pd.DataFrame,
                   drop_st: bool = True,
                   drop_suspended: bool = True,
                   min_liquidity: float = 1000000,
                   min_price: float = 1.0,
                   max_price: float = 1000.0) -> pd.DataFrame:
        """
        数据清洗
        
        Args:
            df: 原始数据
            drop_st: 是否剔除ST股票
            drop_suspended: 是否剔除停牌股票
            min_liquidity: 最小流动性要求（成交额）
            min_price: 最小股价
            max_price: 最大股价
            
        Returns:
            清洗后的数据
        """
        print(f"Original data size: {len(df)}")
        
        cleaned_df = df.copy()
        
        # 剔除ST股票
        if drop_st and 'stock_name' in cleaned_df.columns:
            before_len = len(cleaned_df)
            cleaned_df = cleaned_df[~cleaned_df['stock_name'].str.contains('ST', na=False)]
            print(f"Dropped ST stocks: {before_len - len(cleaned_df)}")
        
        # 剔除停牌股票
        if drop_suspended and 'volume' in cleaned_df.columns:
            before_len = len(cleaned_df)
            cleaned_df = cleaned_df[cleaned_df['volume'] > 0]
            print(f"Dropped suspended stocks: {before_len - len(cleaned_df)}")
        
        # 流动性筛选
        if min_liquidity and 'amount' in cleaned_df.columns:
            before_len = len(cleaned_df)
            cleaned_df = cleaned_df[cleaned_df['amount'] >= min_liquidity]
            print(f"Dropped low liquidity: {before_len - len(cleaned_df)}")
        
        # 价格筛选
        if 'close' in cleaned_df.columns:
            before_len = len(cleaned_df)
            cleaned_df = cleaned_df[
                (cleaned_df['close'] >= min_price) & 
                (cleaned_df['close'] <= max_price)
            ]
            print(f"Dropped out-of-range prices: {before_len - len(cleaned_df)}")
        
        print(f"Cleaned data size: {len(cleaned_df)}")
        
        return cleaned_df
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame,
                             columns: List[str],
                             method: str = 'forward_fill',
                             fill_value: float = 0) -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            df: 数据框
            columns: 需要处理的列
            method: 处理方法 ('forward_fill', 'backward_fill', 'mean', 'median', 'constant')
            fill_value: 常数填充值
            
        Returns:
            处理后的数据
        """
        processed_df = df.copy()
        
        for col in columns:
            if col not in processed_df.columns:
                continue
            
            missing_count = processed_df[col].isna().sum()
            if missing_count == 0:
                continue
            
            print(f"Handling missing values in {col}: {missing_count} missing")
            
            if method == 'forward_fill':
                if 'stock_code' in processed_df.columns:
                    processed_df[col] = processed_df.groupby('stock_code')[col].fillna(method='ffill')
                else:
                    processed_df[col] = processed_df[col].fillna(method='ffill')
                    
            elif method == 'backward_fill':
                if 'stock_code' in processed_df.columns:
                    processed_df[col] = processed_df.groupby('stock_code')[col].fillna(method='bfill')
                else:
                    processed_df[col] = processed_df[col].fillna(method='bfill')
                    
            elif method == 'mean':
                if 'date' in processed_df.columns:
                    # 用同一日期的均值填充
                    processed_df[col] = processed_df.groupby('date')[col].transform(
                        lambda x: x.fillna(x.mean())
                    )
                else:
                    processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
                    
            elif method == 'median':
                if 'date' in processed_df.columns:
                    processed_df[col] = processed_df.groupby('date')[col].transform(
                        lambda x: x.fillna(x.median())
                    )
                else:
                    processed_df[col] = processed_df[col].fillna(processed_df[col].median())
                    
            elif method == 'constant':
                processed_df[col] = processed_df[col].fillna(fill_value)
            
            remaining_missing = processed_df[col].isna().sum()
            print(f"Remaining missing values in {col}: {remaining_missing}")
        
        return processed_df
    
    @staticmethod
    def merge_data(price_df: pd.DataFrame,
                   financial_df: pd.DataFrame = None,
                   market_df: pd.DataFrame = None,
                   on: List[str] = ['stock_code', 'date']) -> pd.DataFrame:
        """
        合并不同来源的数据
        
        Args:
            price_df: 价格数据
            financial_df: 财务数据
            market_df: 市场数据
            on: 合并键
            
        Returns:
            合并后的数据
        """
        merged_df = price_df.copy()
        
        # 合并财务数据
        if financial_df is not None:
            print(f"Merging financial data: {len(financial_df)} records")
            
            # 财务数据通常是季度数据，需要扩展到日度
            if 'report_date' in financial_df.columns:
                # 使用asof merge实现时间对齐
                merged_df = merged_df.sort_values('date')
                financial_df = financial_df.sort_values('report_date')
                
                # 简化处理：直接merge
                merged_df = pd.merge(
                    merged_df,
                    financial_df,
                    left_on=['stock_code'],
                    right_on=['stock_code'],
                    how='left'
                )
            else:
                merged_df = pd.merge(merged_df, financial_df, on=on, how='left')
        
        # 合并市场数据
        if market_df is not None:
            print(f"Merging market data: {len(market_df)} records")
            merged_df = pd.merge(merged_df, market_df, on=on, how='left')
        
        print(f"Merged data size: {len(merged_df)}")
        
        return merged_df
    
    @staticmethod
    def add_industry_info(df: pd.DataFrame, 
                         stock_list: pd.DataFrame) -> pd.DataFrame:
        """
        添加行业信息
        
        Args:
            df: 主数据框
            stock_list: 包含行业信息的股票列表
            
        Returns:
            添加行业信息后的数据
        """
        if 'industry' in stock_list.columns:
            industry_map = dict(zip(stock_list['stock_code'], stock_list['industry']))
            df['industry'] = df['stock_code'].map(industry_map)
            
            print(f"Added industry info. Unique industries: {df['industry'].nunique()}")
        
        return df
    
    @staticmethod
    def calculate_returns(df: pd.DataFrame,
                         periods: List[int] = [1, 5, 20, 60]) -> pd.DataFrame:
        """
        计算收益率
        
        Args:
            df: 数据框（必须包含close列）
            periods: 收益率计算周期
            
        Returns:
            添加收益率列的数据框
        """
        result_df = df.copy()
        
        for period in periods:
            result_df[f'return_{period}d'] = result_df.groupby('stock_code')['close'].pct_change(period)
        
        return result_df
    
    @staticmethod
    def remove_outliers(df: pd.DataFrame,
                       columns: List[str],
                       method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        移除异常值
        
        Args:
            df: 数据框
            columns: 需要处理的列
            method: 方法 ('iqr', 'zscore')
            threshold: 阈值
            
        Returns:
            处理后的数据
        """
        cleaned_df = df.copy()
        
        for col in columns:
            if col not in cleaned_df.columns:
                continue
            
            if method == 'iqr':
                Q1 = cleaned_df[col].quantile(0.25)
                Q3 = cleaned_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                before_len = len(cleaned_df)
                cleaned_df = cleaned_df[
                    (cleaned_df[col] >= lower_bound) & 
                    (cleaned_df[col] <= upper_bound)
                ]
                print(f"Removed {before_len - len(cleaned_df)} outliers in {col} using IQR")
                
            elif method == 'zscore':
                mean = cleaned_df[col].mean()
                std = cleaned_df[col].std()
                z_scores = np.abs((cleaned_df[col] - mean) / std)
                
                before_len = len(cleaned_df)
                cleaned_df = cleaned_df[z_scores < threshold]
                print(f"Removed {before_len - len(cleaned_df)} outliers in {col} using z-score")
        
        return cleaned_df
    
    @staticmethod
    def resample_data(df: pd.DataFrame,
                     freq: str = 'W',
                     agg_dict: Dict = None) -> pd.DataFrame:
        """
        数据重采样（如日度转周度）
        
        Args:
            df: 数据框
            freq: 频率 ('W'周, 'M'月, 'Q'季)
            agg_dict: 聚合字典
            
        Returns:
            重采样后的数据
        """
        if 'date' not in df.columns:
            raise ValueError("DataFrame must have 'date' column")
        
        df = df.set_index('date')
        
        if agg_dict is None:
            # 默认聚合方式
            agg_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'amount': 'sum'
            }
        
        # 按股票分组重采样
        if 'stock_code' in df.columns:
            resampled = df.groupby('stock_code').resample(freq).agg(agg_dict).reset_index()
        else:
            resampled = df.resample(freq).agg(agg_dict).reset_index()
        
        return resampled
    
    @staticmethod
    def get_data_summary(df: pd.DataFrame) -> Dict:
        """
        获取数据摘要统计
        
        Args:
            df: 数据框
            
        Returns:
            统计信息字典
        """
        summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'date_range': (df['date'].min(), df['date'].max()) if 'date' in df.columns else None,
            'unique_stocks': df['stock_code'].nunique() if 'stock_code' in df.columns else None,
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        }
        
        return summary


def test_data_processor():
    """测试数据处理功能"""
    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    test_data = pd.DataFrame({
        'date': dates.tolist() * 3,
        'stock_code': ['000001']*10 + ['000002']*10 + ['000003']*10,
        'close': np.random.rand(30) * 100 + 10,
        'volume': np.random.rand(30) * 1000000,
        'amount': np.random.rand(30) * 10000000
    })
    
    # 添加一些缺失值
    test_data.loc[5, 'close'] = np.nan
    test_data.loc[15, 'volume'] = 0
    
    processor = DataProcessor()
    
    # 测试清洗
    print("Testing clean_data...")
    cleaned = processor.clean_data(test_data, drop_suspended=True, min_liquidity=1000000)
    
    # 测试缺失值处理
    print("\nTesting handle_missing_values...")
    filled = processor.handle_missing_values(cleaned, ['close'], method='forward_fill')
    
    # 测试数据摘要
    print("\nTesting get_data_summary...")
    summary = processor.get_data_summary(filled)
    print("Data Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_data_processor()
