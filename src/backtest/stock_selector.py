"""
选股策略模块
实现多种选股方法：Top N、阈值选股、组合优化
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class StockSelector:
    """选股策略类"""
    
    def __init__(self, 
                 model,
                 n_stocks: int = 50,
                 method: str = 'top_n'):
        """
        初始化选股器
        
        Args:
            model: 训练好的预测模型
            n_stocks: 选股数量
            method: 选股方法 ('top_n', 'threshold', 'portfolio')
        """
        self.model = model
        self.n_stocks = n_stocks
        self.method = method
        print(f"StockSelector initialized: method={method}, n_stocks={n_stocks}")
    
    def select_stocks(self,
                     X: pd.DataFrame,
                     date: str,
                     stock_codes: List[str],
                     **kwargs) -> pd.DataFrame:
        """
        选择股票
        
        Args:
            X: 特征数据
            date: 日期
            stock_codes: 股票代码列表
            **kwargs: 其他参数
            
        Returns:
            选中的股票DataFrame（包含stock_code, score, weight列）
        """
        # 预测得分
        predictions = self.model.predict(X)
        
        # 处理多分类和二分类的不同输出格式
        if len(predictions.shape) == 2:
            # 多分类：取最高类别的概率作为得分
            scores = np.max(predictions, axis=1)
        else:
            # 二分类：直接使用概率
            scores = predictions
        
        # 构建结果DataFrame
        results = pd.DataFrame({
            'date': date,
            'stock_code': stock_codes,
            'score': scores
        })
        
        # 根据方法选股
        if self.method == 'top_n':
            selected = self._select_top_n(results)
        elif self.method == 'threshold':
            threshold = kwargs.get('threshold', 0.6)
            selected = self._select_by_threshold(results, threshold)
        elif self.method == 'portfolio':
            selected = self._select_portfolio(results, X, **kwargs)
        else:
            raise ValueError(f"Unknown selection method: {self.method}")
        
        return selected
    
    def _select_top_n(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Top N选股：选择得分最高的N只股票
        
        Args:
            results: 包含得分的结果DataFrame
            
        Returns:
            选中的股票
        """
        selected = results.nlargest(self.n_stocks, 'score').copy()
        
        # 等权重
        selected['weight'] = 1.0 / len(selected)
        
        return selected
    
    def _select_by_threshold(self, 
                            results: pd.DataFrame,
                            threshold: float) -> pd.DataFrame:
        """
        阈值选股：选择得分超过阈值的股票
        
        Args:
            results: 包含得分的结果DataFrame
            threshold: 得分阈值
            
        Returns:
            选中的股票
        """
        selected = results[results['score'] >= threshold].copy()
        
        # 如果超过最大数量，取得分最高的
        if len(selected) > self.n_stocks:
            selected = selected.nlargest(self.n_stocks, 'score')
        
        # 等权重
        if len(selected) > 0:
            selected['weight'] = 1.0 / len(selected)
        
        return selected
    
    def _select_portfolio(self,
                         results: pd.DataFrame,
                         X: pd.DataFrame,
                         **kwargs) -> pd.DataFrame:
        """
        组合优化选股：基于得分和风险控制选股
        
        Args:
            results: 包含得分的结果DataFrame
            X: 特征数据（用于获取行业等信息）
            **kwargs: 额外参数
            
        Returns:
            选中的股票
        """
        max_industry_weight = kwargs.get('max_industry_weight', 0.3)
        weight_method = kwargs.get('weight_method', 'equal')
        
        # 先选择得分较高的候选池（扩大范围）
        candidate_size = min(self.n_stocks * 3, len(results))
        candidates = results.nlargest(candidate_size, 'score').copy()
        
        # 行业分散（如果有行业信息）
        if 'industry' in X.columns:
            industry_map = dict(zip(X.index, X['industry']))
            candidates['industry'] = candidates.index.map(industry_map)
            
            candidates = self._diversify_by_industry(
                candidates, 
                max_industry_weight=max_industry_weight
            )
        
        # 最终选择
        selected = candidates.head(self.n_stocks).copy()
        
        # 权重分配
        if weight_method == 'equal':
            selected['weight'] = 1.0 / len(selected)
        elif weight_method == 'score_weighted':
            # 基于得分加权
            selected['weight'] = selected['score'] / selected['score'].sum()
        elif weight_method == 'rank_weighted':
            # 基于排名加权
            selected['rank'] = selected['score'].rank(ascending=False)
            selected['weight'] = (len(selected) - selected['rank'] + 1)
            selected['weight'] = selected['weight'] / selected['weight'].sum()
        
        return selected
    
    def _diversify_by_industry(self,
                              candidates: pd.DataFrame,
                              max_industry_weight: float = 0.3) -> pd.DataFrame:
        """
        行业分散化
        
        Args:
            candidates: 候选股票
            max_industry_weight: 单行业最大权重
            
        Returns:
            分散化后的候选股票
        """
        max_per_industry = int(self.n_stocks * max_industry_weight)
        
        selected = []
        industry_counts = {}
        
        # 按得分排序
        candidates = candidates.sort_values('score', ascending=False)
        
        for _, row in candidates.iterrows():
            industry = row.get('industry', 'Unknown')
            
            # 检查该行业是否已达上限
            if industry_counts.get(industry, 0) < max_per_industry:
                selected.append(row)
                industry_counts[industry] = industry_counts.get(industry, 0) + 1
            
            if len(selected) >= self.n_stocks * 2:  # 确保候选池足够大
                break
        
        return pd.DataFrame(selected)
    
    def select_long_short(self,
                         X: pd.DataFrame,
                         date: str,
                         stock_codes: List[str],
                         long_ratio: float = 0.1,
                         short_ratio: float = 0.1) -> Dict[str, pd.DataFrame]:
        """
        多空选股策略
        
        Args:
            X: 特征数据
            date: 日期
            stock_codes: 股票代码列表
            long_ratio: 做多比例（前X%）
            short_ratio: 做空比例（后X%）
            
        Returns:
            字典 {'long': long_stocks, 'short': short_stocks}
        """
        # 预测得分
        scores = self.model.predict(X)
        
        # 构建结果
        results = pd.DataFrame({
            'date': date,
            'stock_code': stock_codes,
            'score': scores
        })
        
        # 计算数量
        n_long = int(len(results) * long_ratio)
        n_short = int(len(results) * short_ratio)
        
        # 选择多头（得分最高）
        long_stocks = results.nlargest(n_long, 'score').copy()
        long_stocks['weight'] = 1.0 / len(long_stocks) if len(long_stocks) > 0 else 0
        long_stocks['position'] = 'long'
        
        # 选择空头（得分最低）
        short_stocks = results.nsmallest(n_short, 'score').copy()
        short_stocks['weight'] = -1.0 / len(short_stocks) if len(short_stocks) > 0 else 0
        short_stocks['position'] = 'short'
        
        return {
            'long': long_stocks,
            'short': short_stocks
        }
    
    def get_selection_summary(self, selected: pd.DataFrame) -> Dict:
        """
        获取选股摘要信息
        
        Args:
            selected: 选中的股票DataFrame
            
        Returns:
            摘要信息字典
        """
        summary = {
            'n_stocks': len(selected),
            'total_weight': selected['weight'].sum(),
            'avg_score': selected['score'].mean(),
            'min_score': selected['score'].min(),
            'max_score': selected['score'].max(),
            'score_std': selected['score'].std()
        }
        
        # 行业分布（如果有）
        if 'industry' in selected.columns:
            summary['industry_distribution'] = selected['industry'].value_counts().to_dict()
        
        return summary


def test_stock_selector():
    """测试选股策略功能"""
    # 创建模拟模型
    class MockModel:
        def predict(self, X):
            # 返回随机得分（模拟预测）
            return np.random.rand(len(X))
    
    # 创建测试数据
    n_stocks = 100
    test_data = pd.DataFrame({
        'feature1': np.random.randn(n_stocks),
        'feature2': np.random.randn(n_stocks),
        'feature3': np.random.randn(n_stocks),
        'industry': [f'Industry_{i%5}' for i in range(n_stocks)]
    })
    
    stock_codes = [f'00{i:04d}' for i in range(n_stocks)]
    date = '2024-01-01'
    
    mock_model = MockModel()
    
    # 测试1: Top N选股
    print("="*60)
    print("Test 1: Top N Selection")
    print("="*60)
    selector = StockSelector(mock_model, n_stocks=20, method='top_n')
    selected = selector.select_stocks(test_data, date, stock_codes)
    print(f"\nSelected {len(selected)} stocks")
    print(selected[['stock_code', 'score', 'weight']].head(10))
    print(f"\nSummary:")
    print(selector.get_selection_summary(selected))
    
    # 测试2: 阈值选股
    print("\n" + "="*60)
    print("Test 2: Threshold Selection")
    print("="*60)
    selector = StockSelector(mock_model, n_stocks=20, method='threshold')
    selected = selector.select_stocks(test_data, date, stock_codes, threshold=0.7)
    print(f"\nSelected {len(selected)} stocks with score > 0.7")
    print(selected[['stock_code', 'score', 'weight']].head(10))
    
    # 测试3: 组合优化选股
    print("\n" + "="*60)
    print("Test 3: Portfolio Selection with Industry Diversification")
    print("="*60)
    selector = StockSelector(mock_model, n_stocks=20, method='portfolio')
    test_data_with_index = test_data.copy()
    test_data_with_index.index = range(len(test_data))
    selected = selector.select_stocks(
        test_data_with_index, date, stock_codes,
        max_industry_weight=0.3,
        weight_method='score_weighted'
    )
    print(f"\nSelected {len(selected)} stocks")
    print(selected[['stock_code', 'score', 'weight']].head(10))
    
    summary = selector.get_selection_summary(selected)
    print(f"\nIndustry Distribution:")
    for industry, count in summary.get('industry_distribution', {}).items():
        print(f"  {industry}: {count}")
    
    # 测试4: 多空选股
    print("\n" + "="*60)
    print("Test 4: Long-Short Selection")
    print("="*60)
    result = selector.select_long_short(test_data, date, stock_codes, long_ratio=0.1, short_ratio=0.1)
    print(f"\nLong positions: {len(result['long'])} stocks")
    print(result['long'][['stock_code', 'score', 'weight']].head(5))
    print(f"\nShort positions: {len(result['short'])} stocks")
    print(result['short'][['stock_code', 'score', 'weight']].head(5))


if __name__ == "__main__":
    test_stock_selector()
