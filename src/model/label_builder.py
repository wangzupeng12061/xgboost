"""
标签构建模块
用于创建机器学习的目标变量（标签）
"""

import pandas as pd
import numpy as np
from typing import Literal, List, Tuple, Optional, Dict


class LabelBuilder:
    """标签构建类"""
    
    def __init__(self, data: pd.DataFrame):
        """
        初始化标签构建器
        
        Args:
            data: 包含价格数据的DataFrame
                 必须包含: stock_code, date, close列
        """
        self.data = data.sort_values(['stock_code', 'date']).reset_index(drop=True)
        print(f"LabelBuilder initialized with {len(data)} rows")
    
    def create_return_label(self, 
                           forward_days: int = 20,
                           label_type: Literal['classification', 'regression'] = 'classification',
                           threshold: float = 0.0,
                           quantiles: List[float] = None) -> pd.DataFrame:
        """
        创建基于收益率的标签
        
        Args:
            forward_days: 前瞻天数
            label_type: 标签类型 ('classification' 或 'regression')
            threshold: 二分类阈值（仅classification+binary模式）
            quantiles: 多分类分位数（如[0.3, 0.7]表示分3组）
            
        Returns:
            带标签的数据
        """
        print(f"\nCreating {label_type} labels with forward_days={forward_days}...")
        df = self.data.copy()
        
        # 计算未来收益率
        df['forward_return'] = df.groupby('stock_code')['close'].pct_change(forward_days).shift(-forward_days)
        
        if label_type == 'classification':
            if quantiles is None:
                # 二分类：涨/跌
                df['label'] = (df['forward_return'] > threshold).astype(int)
                print(f"Binary classification: 0 (下跌) vs 1 (上涨)")
                print(f"Label distribution:\n{df['label'].value_counts()}")
                
            else:
                # 多分类：按分位数分层
                df['label'] = df.groupby('date')['forward_return'].transform(
                    lambda x: self._create_quantile_labels(x, quantiles)
                )
                print(f"Multi-class classification with {len(quantiles)+1} classes")
                print(f"Label distribution:\n{df['label'].value_counts().sort_index()}")
        
        else:  # regression
            # 回归：直接使用收益率
            df['label'] = df['forward_return']
            print(f"Regression: predicting forward returns")
            print(f"Label statistics:\n{df['label'].describe()}")
        
        # 移除没有标签的行（最后N天）
        valid_rows = df['label'].notna().sum()
        df = df[df['label'].notna()]
        print(f"Valid samples: {valid_rows} (removed {len(self.data) - valid_rows} rows without labels)")
        
        return df
    
    @staticmethod
    def _create_quantile_labels(series: pd.Series, quantiles: List[float]) -> pd.Series:
        """根据分位数创建标签"""
        try:
            labels = pd.qcut(
                series, 
                q=[0] + quantiles + [1], 
                labels=range(len(quantiles) + 1),
                duplicates='drop'
            )
            return labels
        except Exception as e:
            # 如果分位数切分失败，返回NaN
            return pd.Series([np.nan] * len(series), index=series.index)
    
    def create_rank_label(self, 
                         forward_days: int = 20,
                         n_groups: int = 5,
                         method: str = 'equal') -> pd.DataFrame:
        """
        创建排名标签（分组）
        
        Args:
            forward_days: 前瞻天数
            n_groups: 分组数量
            method: 分组方法 ('equal' 等距, 'quantile' 等分位数)
            
        Returns:
            带标签的数据
        """
        print(f"\nCreating rank labels with {n_groups} groups...")
        df = self.data.copy()
        
        # 计算未来收益率
        df['forward_return'] = df.groupby('stock_code')['close'].pct_change(forward_days).shift(-forward_days)
        
        # 按日期分组，对每个截面进行排名
        if method == 'quantile':
            df['label'] = df.groupby('date')['forward_return'].transform(
                lambda x: pd.qcut(x, q=n_groups, labels=range(n_groups), duplicates='drop')
            )
        else:  # equal
            df['label'] = df.groupby('date')['forward_return'].transform(
                lambda x: pd.cut(x, bins=n_groups, labels=range(n_groups), duplicates='drop')
            )
        
        print(f"Label distribution:\n{df['label'].value_counts().sort_index()}")
        
        # 移除无效标签
        df = df[df['label'].notna()]
        
        return df
    
    def create_excess_return_label(self,
                                  forward_days: int = 20,
                                  benchmark_col: str = 'benchmark_return',
                                  label_type: str = 'classification') -> pd.DataFrame:
        """
        创建超额收益标签
        
        Args:
            forward_days: 前瞻天数
            benchmark_col: 基准收益率列名
            label_type: 标签类型 ('classification' 或 'regression')
            
        Returns:
            带标签的数据
        """
        print(f"\nCreating excess return labels...")
        df = self.data.copy()
        
        if benchmark_col not in df.columns:
            raise ValueError(f"Benchmark column '{benchmark_col}' not found")
        
        # 计算个股未来收益率
        df['forward_return'] = df.groupby('stock_code')['close'].pct_change(forward_days).shift(-forward_days)
        
        # 计算超额收益
        df['excess_return'] = df['forward_return'] - df[benchmark_col]
        
        if label_type == 'classification':
            # 二分类：跑赢/跑输基准
            df['label'] = (df['excess_return'] > 0).astype(int)
            print(f"Label distribution (0=underperform, 1=outperform):")
            print(df['label'].value_counts())
        else:
            # 回归：预测超额收益大小
            df['label'] = df['excess_return']
            print(f"Excess return statistics:")
            print(df['label'].describe())
        
        df = df[df['label'].notna()]
        
        return df
    
    def create_multi_period_label(self,
                                 periods: List[int] = [5, 10, 20],
                                 aggregation: str = 'mean') -> pd.DataFrame:
        """
        创建多期收益标签（综合多个时间窗口）
        
        Args:
            periods: 前瞻期列表
            aggregation: 聚合方法 ('mean', 'max', 'min', 'weighted')
            
        Returns:
            带标签的数据
        """
        print(f"\nCreating multi-period labels with periods={periods}...")
        df = self.data.copy()
        
        # 计算各期收益
        for period in periods:
            df[f'return_{period}d'] = df.groupby('stock_code')['close'].pct_change(period).shift(-period)
        
        # 聚合
        return_cols = [f'return_{p}d' for p in periods]
        
        if aggregation == 'mean':
            df['label'] = df[return_cols].mean(axis=1)
        elif aggregation == 'max':
            df['label'] = df[return_cols].max(axis=1)
        elif aggregation == 'min':
            df['label'] = df[return_cols].min(axis=1)
        elif aggregation == 'weighted':
            # 权重递减（近期权重更高）
            weights = np.array([1/i for i in range(1, len(periods)+1)])
            weights = weights / weights.sum()
            df['label'] = (df[return_cols] * weights).sum(axis=1)
        
        print(f"Multi-period label statistics:")
        print(df['label'].describe())
        
        df = df[df['label'].notna()]
        
        return df
    
    def create_volatility_adjusted_label(self,
                                        forward_days: int = 20,
                                        vol_window: int = 20) -> pd.DataFrame:
        """
        创建波动率调整后的标签（Sharpe-like）
        
        Args:
            forward_days: 前瞻天数
            vol_window: 波动率计算窗口
            
        Returns:
            带标签的数据
        """
        print(f"\nCreating volatility-adjusted labels...")
        df = self.data.copy()
        
        # 计算历史波动率
        df['daily_return'] = df.groupby('stock_code')['close'].pct_change()
        df['volatility'] = df.groupby('stock_code')['daily_return'].rolling(vol_window).std().reset_index(level=0, drop=True)
        
        # 计算未来收益率
        df['forward_return'] = df.groupby('stock_code')['close'].pct_change(forward_days).shift(-forward_days)
        
        # 波动率调整
        df['label'] = df['forward_return'] / (df['volatility'] * np.sqrt(forward_days))
        df['label'] = df['label'].replace([np.inf, -np.inf], np.nan)
        
        print(f"Volatility-adjusted label statistics:")
        print(df['label'].describe())
        
        df = df[df['label'].notna()]
        
        return df
    
    def add_sample_weights(self,
                          df: pd.DataFrame,
                          method: str = 'time_decay',
                          half_life: int = 60) -> pd.DataFrame:
        """
        添加样本权重（用于模型训练）
        
        Args:
            df: 数据框
            method: 权重方法 ('time_decay', 'uniform', 'volatility')
            half_life: 时间衰减半衰期（天数）
            
        Returns:
            添加权重列的数据
        """
        print(f"\nAdding sample weights using {method} method...")
        
        if method == 'uniform':
            # 均匀权重
            df['sample_weight'] = 1.0
            
        elif method == 'time_decay':
            # 时间衰减权重（越新的样本权重越高）
            df = df.sort_values('date')
            df['days_ago'] = (df['date'].max() - df['date']).dt.days
            df['sample_weight'] = np.exp(-df['days_ago'] / half_life)
            
        elif method == 'volatility':
            # 基于波动率的权重（波动率越小权重越高）
            if 'volatility' not in df.columns:
                df['daily_return'] = df.groupby('stock_code')['close'].pct_change()
                df['volatility'] = df.groupby('stock_code')['daily_return'].rolling(20).std().reset_index(level=0, drop=True)
            
            df['sample_weight'] = 1 / (df['volatility'] + 0.01)
            df['sample_weight'] = df['sample_weight'] / df['sample_weight'].mean()
        
        print(f"Sample weight statistics:")
        print(df['sample_weight'].describe())
        
        return df
    
    def get_label_summary(self, df: pd.DataFrame) -> Dict:
        """
        获取标签统计摘要
        
        Args:
            df: 带标签的数据框
            
        Returns:
            统计信息字典
        """
        if 'label' not in df.columns:
            raise ValueError("No 'label' column found in data")
        
        summary = {
            'total_samples': len(df),
            'label_type': df['label'].dtype,
            'missing_labels': df['label'].isna().sum(),
            'unique_values': df['label'].nunique()
        }
        
        # 如果是分类标签
        if df['label'].dtype in ['int64', 'int32', 'category']:
            summary['class_distribution'] = df['label'].value_counts().to_dict()
            summary['class_balance'] = df['label'].value_counts(normalize=True).to_dict()
        else:
            # 如果是回归标签
            summary['label_stats'] = df['label'].describe().to_dict()
        
        return summary


def test_label_builder():
    """测试标签构建功能"""
    # 创建测试数据
    np.random.seed(42)
    n_dates = 100
    n_stocks = 50
    
    dates = pd.date_range('2024-01-01', periods=n_dates, freq='D')
    
    test_data = []
    for i in range(n_stocks):
        prices = 100 * np.exp(np.cumsum(np.random.randn(n_dates) * 0.02))
        for j, date in enumerate(dates):
            test_data.append({
                'date': date,
                'stock_code': f'00{i:04d}',
                'close': prices[j],
                'benchmark_return': np.random.randn() * 0.01
            })
    
    df = pd.DataFrame(test_data)
    
    # 测试标签构建
    builder = LabelBuilder(df)
    
    # 测试1：二分类标签
    print("\n" + "="*60)
    print("Test 1: Binary Classification Label")
    print("="*60)
    labeled_df = builder.create_return_label(
        forward_days=20,
        label_type='classification',
        threshold=0.0
    )
    print(labeled_df[['date', 'stock_code', 'close', 'forward_return', 'label']].head(10))
    
    # 测试2：多分类标签
    print("\n" + "="*60)
    print("Test 2: Multi-class Label")
    print("="*60)
    labeled_df = builder.create_return_label(
        forward_days=20,
        label_type='classification',
        quantiles=[0.3, 0.7]
    )
    
    # 测试3：回归标签
    print("\n" + "="*60)
    print("Test 3: Regression Label")
    print("="*60)
    labeled_df = builder.create_return_label(
        forward_days=20,
        label_type='regression'
    )
    
    # 测试4：超额收益标签
    print("\n" + "="*60)
    print("Test 4: Excess Return Label")
    print("="*60)
    labeled_df = builder.create_excess_return_label(
        forward_days=20,
        benchmark_col='benchmark_return',
        label_type='classification'
    )
    
    # 测试5：添加样本权重
    print("\n" + "="*60)
    print("Test 5: Sample Weights")
    print("="*60)
    weighted_df = builder.add_sample_weights(labeled_df, method='time_decay')
    
    # 标签摘要
    summary = builder.get_label_summary(weighted_df)
    print("\nLabel Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_label_builder()
