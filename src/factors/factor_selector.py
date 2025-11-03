"""
因子筛选模块
包含IC计算、因子评估、因子筛选等功能
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class FactorSelector:
    """因子筛选类"""
    
    def __init__(self, 
                 factor_data: pd.DataFrame,
                 factor_columns: List[str],
                 forward_return_col: str = 'forward_return'):
        """
        初始化因子筛选器
        
        Args:
            factor_data: 因子数据（必须包含forward_return列）
            factor_columns: 因子列表
            forward_return_col: 前瞻收益率列名
        """
        self.factor_data = factor_data.copy()
        self.factor_columns = factor_columns
        self.forward_return_col = forward_return_col
        self.selected_factors = []
        
        if forward_return_col not in factor_data.columns:
            raise ValueError(f"Forward return column '{forward_return_col}' not found in data")
        
        print(f"FactorSelector initialized with {len(factor_columns)} factors")
    
    def calculate_ic(self, method: str = 'spearman') -> pd.DataFrame:
        """
        计算因子IC值（Information Coefficient）
        
        Args:
            method: 相关系数方法 ('spearman', 'pearson')
            
        Returns:
            IC值DataFrame（每个日期每个因子的IC）
        """
        print(f"\nCalculating IC using {method} method...")
        
        ic_results = []
        dates = self.factor_data['date'].unique()
        
        for date in dates:
            date_data = self.factor_data[self.factor_data['date'] == date]
            
            # 确保有足够的样本
            if len(date_data) < 10:
                continue
            
            ic_values = {'date': date}
            
            for factor in self.factor_columns:
                if factor not in date_data.columns:
                    continue
                
                # 获取有效数据
                valid_data = date_data[[factor, self.forward_return_col]].dropna()
                
                if len(valid_data) < 10:
                    ic_values[factor] = np.nan
                    continue
                
                try:
                    if method == 'spearman':
                        ic, _ = stats.spearmanr(
                            valid_data[factor], 
                            valid_data[self.forward_return_col]
                        )
                    else:  # pearson
                        ic, _ = stats.pearsonr(
                            valid_data[factor], 
                            valid_data[self.forward_return_col]
                        )
                    
                    ic_values[factor] = ic
                    
                except Exception as e:
                    ic_values[factor] = np.nan
            
            ic_results.append(ic_values)
        
        ic_df = pd.DataFrame(ic_results)
        print(f"IC calculated for {len(ic_df)} dates")
        
        return ic_df
    
    def calculate_rank_ic(self) -> pd.DataFrame:
        """
        计算RankIC（基于分组的IC）
        
        Returns:
            RankIC DataFrame
        """
        print("\nCalculating Rank IC...")
        
        rank_ic_results = []
        dates = self.factor_data['date'].unique()
        
        for date in dates:
            date_data = self.factor_data[self.factor_data['date'] == date].copy()
            
            if len(date_data) < 20:
                continue
            
            rank_ic_values = {'date': date}
            
            for factor in self.factor_columns:
                if factor not in date_data.columns:
                    continue
                
                # 因子排名
                date_data[f'{factor}_rank'] = date_data[factor].rank(pct=True)
                # 收益率排名
                date_data['return_rank'] = date_data[self.forward_return_col].rank(pct=True)
                
                # 计算相关性
                valid_data = date_data[[f'{factor}_rank', 'return_rank']].dropna()
                
                if len(valid_data) < 10:
                    rank_ic_values[factor] = np.nan
                    continue
                
                try:
                    ic, _ = stats.spearmanr(valid_data[f'{factor}_rank'], valid_data['return_rank'])
                    rank_ic_values[factor] = ic
                except:
                    rank_ic_values[factor] = np.nan
            
            rank_ic_results.append(rank_ic_values)
        
        return pd.DataFrame(rank_ic_results)
    
    def evaluate_factors(self, ic_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        评估因子表现
        
        Args:
            ic_df: IC数据框（如果为None则重新计算）
            
        Returns:
            因子评估指标DataFrame
        """
        print("\nEvaluating factors...")
        
        if ic_df is None:
            ic_df = self.calculate_ic()
        
        evaluation = []
        
        for factor in self.factor_columns:
            if factor not in ic_df.columns:
                continue
            
            ic_series = ic_df[factor].dropna()
            
            if len(ic_series) == 0:
                continue
            
            # 计算各项指标
            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0  # IC信息比率
            ic_win_rate = (ic_series > 0).sum() / len(ic_series)  # IC胜率
            ic_abs_mean = ic_series.abs().mean()  # IC绝对值均值
            
            evaluation.append({
                'factor': factor,
                'IC_mean': ic_mean,
                'IC_std': ic_std,
                'ICIR': ic_ir,
                'IC_win_rate': ic_win_rate,
                'IC_abs_mean': ic_abs_mean,
                't_stat': ic_mean / (ic_std / np.sqrt(len(ic_series))) if ic_std > 0 else 0
            })
        
        eval_df = pd.DataFrame(evaluation)
        eval_df = eval_df.sort_values('ICIR', ascending=False)
        
        print(f"Evaluated {len(eval_df)} factors")
        
        return eval_df
    
    def select_by_ic(self, 
                     ic_threshold: float = 0.03,
                     icir_threshold: float = 0.5,
                     win_rate_threshold: float = 0.5) -> List[str]:
        """
        基于IC指标筛选因子
        
        Args:
            ic_threshold: IC绝对值均值阈值
            icir_threshold: ICIR阈值
            win_rate_threshold: IC胜率阈值
            
        Returns:
            筛选后的因子列表
        """
        print(f"\nSelecting factors by IC (IC>{ic_threshold}, ICIR>{icir_threshold})...")
        
        evaluation = self.evaluate_factors()
        
        selected = evaluation[
            (evaluation['IC_abs_mean'] >= ic_threshold) &
            (evaluation['ICIR'] >= icir_threshold) &
            (evaluation['IC_win_rate'] >= win_rate_threshold)
        ]
        
        self.selected_factors = selected['factor'].tolist()
        
        print(f"Selected {len(self.selected_factors)} factors:")
        print(selected[['factor', 'IC_mean', 'ICIR', 'IC_win_rate']].to_string(index=False))
        
        return self.selected_factors
    
    def calculate_factor_correlation(self, factors: List[str] = None) -> pd.DataFrame:
        """
        计算因子相关性矩阵
        
        Args:
            factors: 因子列表（默认使用全部因子）
            
        Returns:
            因子相关性矩阵
        """
        if factors is None:
            factors = self.factor_columns
        
        print(f"\nCalculating correlation matrix for {len(factors)} factors...")
        
        # 提取因子数据
        factor_data = self.factor_data[factors]
        
        # 计算相关性
        corr_matrix = factor_data.corr(method='spearman')
        
        return corr_matrix
    
    def remove_correlated_factors(self, 
                                  correlation_threshold: float = 0.7,
                                  factors: List[str] = None) -> List[str]:
        """
        去除高相关因子
        
        Args:
            correlation_threshold: 相关系数阈值
            factors: 要处理的因子列表（默认使用全部因子）
            
        Returns:
            去相关后的因子列表
        """
        if factors is None:
            factors = self.factor_columns if not self.selected_factors else self.selected_factors
        
        print(f"\nRemoving correlated factors (threshold={correlation_threshold})...")
        
        # 计算相关矩阵
        corr_matrix = self.calculate_factor_correlation(factors)
        
        # 获取因子评估结果，用于排序
        evaluation = self.evaluate_factors()
        factor_scores = dict(zip(evaluation['factor'], evaluation['ICIR']))
        
        # 去相关算法
        selected = []
        remaining = factors.copy()
        
        while remaining:
            # 选择ICIR最高的因子
            best_factor = max(remaining, key=lambda x: factor_scores.get(x, 0))
            selected.append(best_factor)
            remaining.remove(best_factor)
            
            if not remaining:
                break
            
            # 找出与该因子高相关的其他因子
            high_corr_factors = []
            for factor in remaining:
                if abs(corr_matrix.loc[best_factor, factor]) > correlation_threshold:
                    high_corr_factors.append(factor)
            
            # 移除高相关因子
            for factor in high_corr_factors:
                if factor in remaining:
                    remaining.remove(factor)
        
        self.selected_factors = selected
        
        print(f"Selected {len(selected)} uncorrelated factors:")
        print(selected)
        
        return selected
    
    def get_factor_groups(self, n_groups: int = 5) -> Dict[str, pd.DataFrame]:
        """
        根据因子值分组，分析收益分布
        
        Args:
            n_groups: 分组数量
            
        Returns:
            各因子的分组收益统计
        """
        print(f"\nAnalyzing factor groups (n_groups={n_groups})...")
        
        group_results = {}
        
        for factor in self.factor_columns:
            if factor not in self.factor_data.columns:
                continue
            
            # 按日期分组
            group_returns = []
            
            for date in self.factor_data['date'].unique():
                date_data = self.factor_data[
                    self.factor_data['date'] == date
                ][[factor, self.forward_return_col]].dropna()
                
                if len(date_data) < n_groups * 2:
                    continue
                
                # 分组
                date_data['group'] = pd.qcut(
                    date_data[factor], 
                    q=n_groups, 
                    labels=range(1, n_groups + 1),
                    duplicates='drop'
                )
                
                # 计算各组平均收益
                group_mean = date_data.groupby('group')[self.forward_return_col].mean()
                
                for group_id, ret in group_mean.items():
                    group_returns.append({
                        'date': date,
                        'group': group_id,
                        'return': ret
                    })
            
            if group_returns:
                group_df = pd.DataFrame(group_returns)
                # 计算各组的平均收益
                summary = group_df.groupby('group')['return'].agg(['mean', 'std', 'count'])
                summary['sharpe'] = summary['mean'] / summary['std']
                
                group_results[factor] = summary
        
        return group_results
    
    def get_ic_decay(self, max_periods: int = 10) -> pd.DataFrame:
        """
        计算IC衰减（不同前瞻期的IC）
        
        Args:
            max_periods: 最大前瞻期
            
        Returns:
            IC衰减DataFrame
        """
        print(f"\nCalculating IC decay (max_periods={max_periods})...")
        
        decay_results = []
        
        for period in range(1, max_periods + 1):
            # 计算不同期数的前瞻收益
            temp_data = self.factor_data.copy()
            temp_data[f'forward_return_{period}'] = temp_data.groupby('stock_code')['close'].pct_change(period).shift(-period)
            
            # 计算IC
            ic_list = []
            for date in temp_data['date'].unique():
                date_data = temp_data[temp_data['date'] == date]
                
                for factor in self.factor_columns:
                    if factor not in date_data.columns:
                        continue
                    
                    valid_data = date_data[[factor, f'forward_return_{period}']].dropna()
                    
                    if len(valid_data) < 10:
                        continue
                    
                    try:
                        ic, _ = stats.spearmanr(valid_data[factor], valid_data[f'forward_return_{period}'])
                        ic_list.append({'factor': factor, 'period': period, 'IC': ic})
                    except:
                        continue
            
            if ic_list:
                period_df = pd.DataFrame(ic_list)
                period_mean = period_df.groupby('factor')['IC'].mean().to_dict()
                
                for factor, ic in period_mean.items():
                    decay_results.append({
                        'factor': factor,
                        'period': period,
                        'IC_mean': ic
                    })
        
        return pd.DataFrame(decay_results)


def test_factor_selector():
    """测试因子筛选功能"""
    # 创建测试数据
    np.random.seed(42)
    n_dates = 50
    n_stocks = 100
    
    dates = pd.date_range('2024-01-01', periods=n_dates, freq='D')
    
    test_data = []
    for date in dates:
        for i in range(n_stocks):
            # 创建有预测性的因子
            factor1 = np.random.randn()
            factor2 = np.random.randn()
            factor3 = np.random.randn()
            
            # forward_return与factor1正相关
            forward_return = 0.5 * factor1 + 0.3 * factor2 + np.random.randn() * 0.5
            
            test_data.append({
                'date': date,
                'stock_code': f'00{i:04d}',
                'close': np.random.rand() * 100 + 10,
                'factor1': factor1,
                'factor2': factor2,
                'factor3': factor3,
                'forward_return': forward_return
            })
    
    df = pd.DataFrame(test_data)
    
    # 测试因子筛选
    selector = FactorSelector(df, ['factor1', 'factor2', 'factor3'])
    
    # 计算IC
    ic_df = selector.calculate_ic()
    print("\nIC Statistics:")
    print(ic_df[['factor1', 'factor2', 'factor3']].describe())
    
    # 评估因子
    evaluation = selector.evaluate_factors(ic_df)
    print("\nFactor Evaluation:")
    print(evaluation)
    
    # 筛选因子
    selected = selector.select_by_ic(ic_threshold=0.02, icir_threshold=0.3)
    
    # 去相关
    final_selected = selector.remove_correlated_factors(correlation_threshold=0.7)
    
    print(f"\nFinal selected factors: {final_selected}")


if __name__ == "__main__":
    test_factor_selector()
