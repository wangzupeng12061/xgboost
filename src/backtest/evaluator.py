"""
绩效评估模块
计算各种绩效指标和风险指标
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from scipy import stats


class PerformanceEvaluator:
    """绩效评估类"""
    
    def __init__(self,
                 portfolio_values: pd.Series,
                 benchmark_values: pd.Series = None,
                 risk_free_rate: float = 0.03):
        """
        初始化评估器
        
        Args:
            portfolio_values: 组合净值序列（index为日期）
            benchmark_values: 基准净值序列
            risk_free_rate: 无风险利率（年化）
        """
        self.portfolio_values = portfolio_values.sort_index()
        self.benchmark_values = benchmark_values.sort_index() if benchmark_values is not None else None
        self.risk_free_rate = risk_free_rate
        
        # 计算收益率
        self.returns = self.portfolio_values.pct_change().dropna()
        
        if self.benchmark_values is not None:
            self.benchmark_returns = self.benchmark_values.pct_change().dropna()
        else:
            self.benchmark_returns = None
        
        print(f"PerformanceEvaluator initialized")
        print(f"  Period: {self.portfolio_values.index[0]} to {self.portfolio_values.index[-1]}")
        print(f"  Trading days: {len(self.portfolio_values)}")
    
    def calculate_all_metrics(self) -> Dict[str, float]:
        """
        计算所有绩效指标
        
        Returns:
            指标字典
        """
        metrics = {}
        
        # 收益指标
        metrics['total_return'] = self.total_return()
        metrics['annual_return'] = self.annual_return()
        metrics['daily_return_mean'] = self.returns.mean() * 100
        
        # 风险指标
        metrics['volatility'] = self.volatility()
        metrics['downside_volatility'] = self.downside_volatility()
        metrics['max_drawdown'] = self.max_drawdown()
        metrics['max_drawdown_duration'] = self.max_drawdown_duration()
        
        # 风险调整收益
        metrics['sharpe_ratio'] = self.sharpe_ratio()
        metrics['sortino_ratio'] = self.sortino_ratio()
        metrics['calmar_ratio'] = self.calmar_ratio()
        metrics['omega_ratio'] = self.omega_ratio()
        
        # 相对指标（如果有基准）
        if self.benchmark_returns is not None:
            metrics['alpha'] = self.alpha()
            metrics['beta'] = self.beta()
            metrics['information_ratio'] = self.information_ratio()
            metrics['tracking_error'] = self.tracking_error()
            metrics['active_return'] = self.active_return()
            metrics['up_capture'] = self.up_capture_ratio()
            metrics['down_capture'] = self.down_capture_ratio()
        
        # 其他指标
        metrics['win_rate'] = self.win_rate()
        metrics['profit_loss_ratio'] = self.profit_loss_ratio()
        metrics['var_95'] = self.value_at_risk(0.95)
        metrics['cvar_95'] = self.conditional_var(0.95)
        
        return metrics
    
    # ========== 收益指标 ==========
    
    def total_return(self) -> float:
        """累计收益率"""
        return (self.portfolio_values.iloc[-1] / self.portfolio_values.iloc[0] - 1) * 100
    
    def annual_return(self) -> float:
        """年化收益率"""
        days = len(self.portfolio_values)
        years = days / 252
        total_return = self.total_return() / 100
        return (np.power(1 + total_return, 1 / years) - 1) * 100
    
    # ========== 风险指标 ==========
    
    def volatility(self) -> float:
        """年化波动率"""
        return self.returns.std() * np.sqrt(252) * 100
    
    def downside_volatility(self, threshold: float = 0) -> float:
        """下行波动率（年化）"""
        downside_returns = self.returns[self.returns < threshold]
        return downside_returns.std() * np.sqrt(252) * 100
    
    def max_drawdown(self) -> float:
        """最大回撤"""
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min() * 100
    
    def max_drawdown_duration(self) -> int:
        """最大回撤持续天数"""
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # 找出所有回撤期
        is_drawdown = drawdown < 0
        drawdown_periods = []
        start = None
        
        for i, dd in enumerate(is_drawdown):
            if dd and start is None:
                start = i
            elif not dd and start is not None:
                drawdown_periods.append(i - start)
                start = None
        
        if start is not None:
            drawdown_periods.append(len(is_drawdown) - start)
        
        return max(drawdown_periods) if drawdown_periods else 0
    
    # ========== 风险调整收益 ==========
    
    def sharpe_ratio(self) -> float:
        """夏普比率"""
        excess_return = self.annual_return() / 100 - self.risk_free_rate
        volatility = self.volatility() / 100
        return excess_return / volatility if volatility != 0 else 0
    
    def sortino_ratio(self, threshold: float = 0) -> float:
        """索提诺比率（使用下行波动率）"""
        annual_return = self.annual_return() / 100
        downside_vol = self.downside_volatility(threshold) / 100
        return (annual_return - self.risk_free_rate) / downside_vol if downside_vol != 0 else 0
    
    def calmar_ratio(self) -> float:
        """卡玛比率（年化收益/最大回撤）"""
        annual_return = self.annual_return()
        max_dd = abs(self.max_drawdown())
        return annual_return / max_dd if max_dd != 0 else 0
    
    def omega_ratio(self, threshold: float = 0) -> float:
        """Omega比率"""
        returns_above = self.returns[self.returns > threshold].sum()
        returns_below = abs(self.returns[self.returns < threshold].sum())
        return returns_above / returns_below if returns_below != 0 else 0
    
    # ========== 相对指标 ==========
    
    def alpha(self) -> float:
        """Alpha（相对基准的超额收益）"""
        if self.benchmark_returns is None:
            return np.nan
        
        beta = self.beta()
        portfolio_return = self.annual_return() / 100
        
        benchmark_return = (
            self.benchmark_values.iloc[-1] / self.benchmark_values.iloc[0] - 1
        )
        days = len(self.benchmark_values)
        benchmark_annual = np.power(1 + benchmark_return, 252 / days) - 1
        
        alpha = portfolio_return - (self.risk_free_rate + beta * (benchmark_annual - self.risk_free_rate))
        return alpha * 100
    
    def beta(self) -> float:
        """Beta（系统风险）"""
        if self.benchmark_returns is None:
            return np.nan
        
        # 对齐数据
        aligned = pd.DataFrame({
            'portfolio': self.returns,
            'benchmark': self.benchmark_returns
        }).dropna()
        
        if len(aligned) < 2:
            return np.nan
        
        covariance = aligned['portfolio'].cov(aligned['benchmark'])
        benchmark_variance = aligned['benchmark'].var()
        
        return covariance / benchmark_variance if benchmark_variance != 0 else np.nan
    
    def information_ratio(self) -> float:
        """信息比率"""
        if self.benchmark_returns is None:
            return np.nan
        
        # 对齐数据
        aligned = pd.DataFrame({
            'portfolio': self.returns,
            'benchmark': self.benchmark_returns
        }).dropna()
        
        excess_returns = aligned['portfolio'] - aligned['benchmark']
        
        if len(excess_returns) < 2:
            return np.nan
        
        return (excess_returns.mean() / excess_returns.std() * np.sqrt(252)) if excess_returns.std() != 0 else 0
    
    def tracking_error(self) -> float:
        """跟踪误差（年化）"""
        if self.benchmark_returns is None:
            return np.nan
        
        aligned = pd.DataFrame({
            'portfolio': self.returns,
            'benchmark': self.benchmark_returns
        }).dropna()
        
        excess_returns = aligned['portfolio'] - aligned['benchmark']
        return excess_returns.std() * np.sqrt(252) * 100
    
    def active_return(self) -> float:
        """主动收益（年化）"""
        if self.benchmark_returns is None:
            return np.nan
        
        portfolio_annual = self.annual_return()
        
        benchmark_return = (
            self.benchmark_values.iloc[-1] / self.benchmark_values.iloc[0] - 1
        )
        days = len(self.benchmark_values)
        benchmark_annual = np.power(1 + benchmark_return, 252 / days) - 1
        
        return (portfolio_annual / 100 - benchmark_annual) * 100
    
    def up_capture_ratio(self) -> float:
        """上行捕获率"""
        if self.benchmark_returns is None:
            return np.nan
        
        aligned = pd.DataFrame({
            'portfolio': self.returns,
            'benchmark': self.benchmark_returns
        }).dropna()
        
        up_markets = aligned[aligned['benchmark'] > 0]
        
        if len(up_markets) == 0:
            return np.nan
        
        portfolio_up = up_markets['portfolio'].mean()
        benchmark_up = up_markets['benchmark'].mean()
        
        return (portfolio_up / benchmark_up * 100) if benchmark_up != 0 else np.nan
    
    def down_capture_ratio(self) -> float:
        """下行捕获率"""
        if self.benchmark_returns is None:
            return np.nan
        
        aligned = pd.DataFrame({
            'portfolio': self.returns,
            'benchmark': self.benchmark_returns
        }).dropna()
        
        down_markets = aligned[aligned['benchmark'] < 0]
        
        if len(down_markets) == 0:
            return np.nan
        
        portfolio_down = down_markets['portfolio'].mean()
        benchmark_down = down_markets['benchmark'].mean()
        
        return (portfolio_down / benchmark_down * 100) if benchmark_down != 0 else np.nan
    
    # ========== 其他指标 ==========
    
    def win_rate(self) -> float:
        """胜率（盈利交易日占比）"""
        return (self.returns > 0).sum() / len(self.returns) * 100
    
    def profit_loss_ratio(self) -> float:
        """盈亏比（平均盈利/平均亏损）"""
        wins = self.returns[self.returns > 0]
        losses = self.returns[self.returns < 0]
        
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
        
        return avg_win / avg_loss if avg_loss != 0 else np.nan
    
    def value_at_risk(self, confidence: float = 0.95) -> float:
        """VaR（风险价值）"""
        return np.percentile(self.returns, (1 - confidence) * 100) * 100
    
    def conditional_var(self, confidence: float = 0.95) -> float:
        """CVaR（条件风险价值）"""
        var = self.value_at_risk(confidence) / 100
        return self.returns[self.returns <= var].mean() * 100
    
    # ========== 报告生成 ==========
    
    def generate_report(self) -> str:
        """生成绩效报告"""
        metrics = self.calculate_all_metrics()
        
        report = "\n" + "="*80 + "\n"
        report += "绩效评估报告\n"
        report += "="*80 + "\n\n"
        
        report += "📊 收益指标\n"
        report += "-"*80 + "\n"
        report += f"  累计收益率:        {metrics['total_return']:>10.2f}%\n"
        report += f"  年化收益率:        {metrics['annual_return']:>10.2f}%\n"
        report += f"  日均收益率:        {metrics['daily_return_mean']:>10.4f}%\n"
        
        report += "\n📉 风险指标\n"
        report += "-"*80 + "\n"
        report += f"  年化波动率:        {metrics['volatility']:>10.2f}%\n"
        report += f"  下行波动率:        {metrics['downside_volatility']:>10.2f}%\n"
        report += f"  最大回撤:          {metrics['max_drawdown']:>10.2f}%\n"
        report += f"  最大回撤天数:      {metrics['max_drawdown_duration']:>10.0f} 天\n"
        report += f"  95% VaR:          {metrics['var_95']:>10.2f}%\n"
        report += f"  95% CVaR:         {metrics['cvar_95']:>10.2f}%\n"
        
        report += "\n⚖️  风险调整收益\n"
        report += "-"*80 + "\n"
        report += f"  夏普比率:          {metrics['sharpe_ratio']:>10.4f}\n"
        report += f"  索提诺比率:        {metrics['sortino_ratio']:>10.4f}\n"
        report += f"  卡玛比率:          {metrics['calmar_ratio']:>10.4f}\n"
        report += f"  Omega比率:        {metrics['omega_ratio']:>10.4f}\n"
        
        if self.benchmark_returns is not None:
            report += "\n📈 相对基准指标\n"
            report += "-"*80 + "\n"
            report += f"  Alpha:            {metrics['alpha']:>10.2f}%\n"
            report += f"  Beta:             {metrics['beta']:>10.4f}\n"
            report += f"  信息比率:          {metrics['information_ratio']:>10.4f}\n"
            report += f"  跟踪误差:          {metrics['tracking_error']:>10.2f}%\n"
            report += f"  主动收益:          {metrics['active_return']:>10.2f}%\n"
            report += f"  上行捕获率:        {metrics['up_capture']:>10.2f}%\n"
            report += f"  下行捕获率:        {metrics['down_capture']:>10.2f}%\n"
        
        report += "\n🎯 交易统计\n"
        report += "-"*80 + "\n"
        report += f"  胜率:              {metrics['win_rate']:>10.2f}%\n"
        report += f"  盈亏比:            {metrics['profit_loss_ratio']:>10.4f}\n"
        
        report += "\n" + "="*80 + "\n"
        
        return report


def test_evaluator():
    """测试绩效评估功能"""
    print("="*60)
    print("测试绩效评估")
    print("="*60)
    
    # 创建模拟数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=252, freq='D')
    
    # 模拟组合净值（有上涨趋势）
    returns = np.random.randn(252) * 0.02 + 0.001
    portfolio_values = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)
    
    # 模拟基准净值
    benchmark_returns = np.random.randn(252) * 0.015 + 0.0005
    benchmark_values = pd.Series(100 * np.exp(np.cumsum(benchmark_returns)), index=dates)
    
    # 评估
    evaluator = PerformanceEvaluator(
        portfolio_values=portfolio_values,
        benchmark_values=benchmark_values,
        risk_free_rate=0.03
    )
    
    # 打印报告
    print(evaluator.generate_report())
    
    # 获取所有指标
    metrics = evaluator.calculate_all_metrics()
    
    print("\n所有指标:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    test_evaluator()
