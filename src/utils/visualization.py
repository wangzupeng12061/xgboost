"""
可视化模块
绘制净值曲线、回撤、因子分析等图表
"""

import pandas as pd
import numpy as np
import os
import platform

# 先加载字体，再设置后端
import matplotlib.font_manager as fm

# 显式加载中文字体文件
system = platform.system()
if system == 'Darwin':
    font_path = '/System/Library/Fonts/Supplemental/Songti.ttc'
    if os.path.exists(font_path):
        try:
            fm.fontManager.addfont(font_path)
            print(f"✓ 预加载字体: {font_path}")
        except:
            pass

# 现在设置后端
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# 创建中文字体属性对象
def get_chinese_font():
    """获取中文字体属性"""
    system = platform.system()
    
    if system == 'Darwin':
        font_paths = [
            '/System/Library/Fonts/Supplemental/Songti.ttc',
            '/System/Library/Fonts/STHeiti Medium.ttc'
        ]
    elif system == 'Windows':
        font_paths = [
            'C:/Windows/Fonts/msyh.ttc',
            'C:/Windows/Fonts/simhei.ttf'
        ]
    else:
        font_paths = []
    
    for path in font_paths:
        if os.path.exists(path):
            return font_manager.FontProperties(fname=path)
    
    # 如果找不到文件，尝试使用字体名称
    return font_manager.FontProperties(family=['Songti SC', 'SimHei', 'WenQuanYi Micro Hei'])

# 全局字体属性
CHINESE_FONT = get_chinese_font()

# 设置全局字体
plt.rcParams['font.sans-serif'] = ['Songti SC', 'STSong', 'Heiti SC', 'SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

print(f"✓ 中文字体已配置: {plt.rcParams['font.sans-serif'][0]}")

# 设置样式
sns.set_style("whitegrid")
plt.rcParams['figure.autolayout'] = True


def plot_equity_curve(portfolio_values: pd.Series,
                     benchmark_values: pd.Series = None,
                     title: str = '净值曲线',
                     figsize: Tuple[int, int] = (14, 6),
                     save_path: str = None):
    """
    绘制净值曲线
    
    Args:
        portfolio_values: 组合净值
        benchmark_values: 基准净值
        title: 图表标题
        figsize: 图片大小
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 归一化
    portfolio_norm = portfolio_values / portfolio_values.iloc[0] * 100
    
    # 绘制组合曲线
    ax.plot(portfolio_norm.index, portfolio_norm.values,
            label='策略组合', linewidth=2, color='#E74C3C')
    
    # 绘制基准曲线
    if benchmark_values is not None and len(benchmark_values) > 0:
        # 使用线性插值对齐基准数据到组合日期，避免横跳
        benchmark_norm = benchmark_values / benchmark_values.iloc[0] * 100
        benchmark_aligned = benchmark_norm.reindex(
            portfolio_norm.index.union(benchmark_norm.index)
        ).interpolate(method='time').reindex(portfolio_norm.index)
        ax.plot(benchmark_aligned.index, benchmark_aligned.values,
                label='基准指数', linewidth=2, color='#3498DB', alpha=0.7)
    
    ax.set_title(title, fontproperties=CHINESE_FONT, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('日期', fontproperties=CHINESE_FONT, fontsize=12)
    ax.set_ylabel('净值（归一化）', fontproperties=CHINESE_FONT, fontsize=12)
    ax.legend(fontsize=12, loc='upper left', prop=CHINESE_FONT)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    
    plt.close()


def plot_drawdown(portfolio_values: pd.Series,
                 benchmark_values: pd.Series = None,
                 title: str = '回撤曲线',
                 figsize: Tuple[int, int] = (14, 6),
                 save_path: str = None):
    """
    绘制回撤曲线
    
    Args:
        portfolio_values: 组合净值
        benchmark_values: 基准净值
        title: 图表标题
        figsize: 图片大小
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 计算回撤
    portfolio_norm = portfolio_values / portfolio_values.iloc[0]
    portfolio_cummax = portfolio_norm.expanding().max()
    portfolio_drawdown = (portfolio_norm - portfolio_cummax) / portfolio_cummax * 100
    
    # 绘制组合回撤
    ax.fill_between(portfolio_drawdown.index, 0, portfolio_drawdown.values,
                     label='策略组合', color='#E74C3C', alpha=0.5)
    
    # 绘制基准回撤
    if benchmark_values is not None and len(benchmark_values) > 0:
        # 使用插值对齐基准数据
        benchmark_norm = benchmark_values / benchmark_values.iloc[0]
        benchmark_aligned = benchmark_norm.reindex(
            portfolio_norm.index.union(benchmark_norm.index)
        ).interpolate(method='time').reindex(portfolio_norm.index)
        benchmark_cummax = benchmark_aligned.expanding().max()
        benchmark_drawdown = (benchmark_aligned - benchmark_cummax) / benchmark_cummax * 100
        
        ax.fill_between(benchmark_drawdown.index, 0, benchmark_drawdown.values,
                         label='基准指数', color='#3498DB', alpha=0.3)
    
    ax.set_title(title, fontproperties=CHINESE_FONT, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('日期', fontproperties=CHINESE_FONT, fontsize=12)
    ax.set_ylabel('回撤 (%)', fontproperties=CHINESE_FONT, fontsize=12)
    ax.legend(fontsize=12, loc='lower left', prop=CHINESE_FONT)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_returns_distribution(portfolio_values: pd.Series,
                              benchmark_values: pd.Series = None,
                              figsize: Tuple[int, int] = (14, 5),
                              save_path: str = None):
    """
    绘制收益率分布
    
    Args:
        portfolio_values: 组合净值
        benchmark_values: 基准净值
        figsize: 图片大小
        save_path: 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 计算收益率
    portfolio_returns = portfolio_values.pct_change().dropna() * 100
    
    # 直方图
    ax1.hist(portfolio_returns, bins=50, alpha=0.7, label='策略组合',
             color='#E74C3C', edgecolor='black')
    
    if benchmark_values is not None and len(benchmark_values) > 0:
        # 使用插值对齐基准数据
        benchmark_aligned = benchmark_values.reindex(
            portfolio_values.index.union(benchmark_values.index)
        ).interpolate(method='time').reindex(portfolio_values.index)
        benchmark_returns = benchmark_aligned.pct_change().dropna() * 100
        ax1.hist(benchmark_returns, bins=50, alpha=0.5, label='基准指数',
                 color='#3498DB', edgecolor='black')
    
    ax1.set_title('日收益率分布', fontproperties=CHINESE_FONT, fontsize=14, fontweight='bold')
    ax1.set_xlabel('收益率 (%)', fontproperties=CHINESE_FONT, fontsize=12)
    ax1.set_ylabel('频数', fontproperties=CHINESE_FONT, fontsize=12)
    ax1.legend(fontsize=10, prop=CHINESE_FONT)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # 箱线图
    data_to_plot = [portfolio_returns]
    labels = ['策略组合']
    
    if benchmark_values is not None and len(benchmark_values) > 0:
        # 使用已经对齐的benchmark_returns（如果存在）
        if 'benchmark_returns' in locals():
            data_to_plot.append(benchmark_returns)
            labels.append('基准指数')
    
    bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
    
    # 设置颜色
    colors = ['#E74C3C', '#3498DB']
    for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_title('收益率箱线图', fontproperties=CHINESE_FONT, fontsize=14, fontweight='bold')
    ax2.set_xticks(range(1, len(labels) + 1))
    ax2.set_xticklabels(labels, fontproperties=CHINESE_FONT, fontsize=10)
    ax2.set_ylabel('收益率 (%)', fontproperties=CHINESE_FONT, fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_rolling_metrics(portfolio_values: pd.Series,
                         window: int = 60,
                         figsize: Tuple[int, int] = (14, 10),
                         save_path: str = None):
    """
    绘制滚动指标
    
    Args:
        portfolio_values: 组合净值
        window: 滚动窗口
        figsize: 图片大小
        save_path: 保存路径
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    
    # 计算收益率
    returns = portfolio_values.pct_change().dropna()
    
    # 滚动收益率
    rolling_return = returns.rolling(window).sum() * 100
    axes[0].plot(rolling_return.index, rolling_return.values,
                 linewidth=2, color='#E74C3C')
    axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axes[0].set_title(f'{window}日滚动收益率', fontproperties=CHINESE_FONT, fontsize=14, fontweight='bold')
    axes[0].set_ylabel('收益率 (%)', fontproperties=CHINESE_FONT, fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # 滚动波动率
    rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100
    axes[1].plot(rolling_vol.index, rolling_vol.values,
                 linewidth=2, color='#3498DB')
    axes[1].set_title(f'{window}日滚动波动率（年化）', fontproperties=CHINESE_FONT, fontsize=14, fontweight='bold')
    axes[1].set_ylabel('波动率 (%)', fontproperties=CHINESE_FONT, fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # 滚动夏普比率
    rolling_sharpe = (rolling_return / 100) / (rolling_vol / 100)
    axes[2].plot(rolling_sharpe.index, rolling_sharpe.values,
                 linewidth=2, color='#27AE60')
    axes[2].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axes[2].set_title(f'{window}日滚动夏普比率', fontproperties=CHINESE_FONT, fontsize=14, fontweight='bold')
    axes[2].set_xlabel('日期', fontproperties=CHINESE_FONT, fontsize=12)
    axes[2].set_ylabel('夏普比率', fontproperties=CHINESE_FONT, fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_monthly_returns(portfolio_values: pd.Series,
                        figsize: Tuple[int, int] = (14, 8),
                        save_path: str = None):
    """
    绘制月度收益热力图
    
    Args:
        portfolio_values: 组合净值
        figsize: 图片大小
        save_path: 保存路径
    """
    # 计算月度收益
    monthly = portfolio_values.resample('M').last().pct_change() * 100
    
    # 重塑为年-月矩阵
    monthly_data = pd.DataFrame({
        'return': monthly.values,
        'year': monthly.index.year,
        'month': monthly.index.month
    })
    
    pivot_table = monthly_data.pivot(index='year', columns='month', values='return')
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=figsize)
    
    heatmap = sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='RdYlGn',
                          center=0, ax=ax, cbar_kws={'label': '收益率 (%)'})
    if heatmap.collections:
        cbar = heatmap.collections[0].colorbar
        cbar.ax.set_ylabel('收益率 (%)', fontproperties=CHINESE_FONT, fontsize=12)
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontproperties(CHINESE_FONT)
        cbar.ax.tick_params(labelsize=10)
    
    ax.set_title('月度收益率热力图', fontproperties=CHINESE_FONT, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('月份', fontproperties=CHINESE_FONT, fontsize=12)
    ax.set_ylabel('年份', fontproperties=CHINESE_FONT, fontsize=12)
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(CHINESE_FONT)
        tick.set_fontsize(10)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(CHINESE_FONT)
        tick.set_fontsize(10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_feature_importance(importance_df: pd.DataFrame,
                           top_n: int = 20,
                           figsize: Tuple[int, int] = (12, 8),
                           save_path: str = None):
    """
    绘制因子重要性
    
    Args:
        importance_df: 因子重要性DataFrame (包含feature和importance列)
        top_n: 显示前N个因子
        figsize: 图片大小
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    top_features = importance_df.head(top_n)
    
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_features)))
    
    ax.barh(range(len(top_features)), top_features['importance'].values,
            color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values, fontproperties=CHINESE_FONT, fontsize=10)
    ax.invert_yaxis()
    
    ax.set_title(f'Top {top_n} 因子重要性', fontproperties=CHINESE_FONT, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('重要性得分', fontproperties=CHINESE_FONT, fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_ic_analysis(ic_df: pd.DataFrame,
                    factors: List[str] = None,
                    figsize: Tuple[int, int] = (14, 10),
                    save_path: str = None):
    """
    绘制IC分析图
    
    Args:
        ic_df: IC DataFrame (包含date列和各因子IC列)
        factors: 要绘制的因子列表
        figsize: 图片大小
        save_path: 保存路径
    """
    if factors is None:
        factors = [col for col in ic_df.columns if col != 'date'][:6]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # IC时序图
    for factor in factors:
        if factor in ic_df.columns:
            ax1.plot(pd.to_datetime(ic_df['date']), ic_df[factor],
                     label=factor, alpha=0.7, linewidth=1.5)
    
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_title('因子IC时序图', fontproperties=CHINESE_FONT, fontsize=14, fontweight='bold')
    ax1.set_xlabel('日期', fontproperties=CHINESE_FONT, fontsize=12)
    ax1.set_ylabel('IC值', fontproperties=CHINESE_FONT, fontsize=12)
    ax1.legend(fontsize=10, loc='best', ncol=2, prop=CHINESE_FONT)
    ax1.grid(True, alpha=0.3)
    
    # IC累计图
    ic_cumsum = ic_df[factors].cumsum()
    
    for i, factor in enumerate(factors):
        if factor in ic_cumsum.columns:
            ax2.plot(pd.to_datetime(ic_df['date']), ic_cumsum[factor],
                     label=factor, alpha=0.7, linewidth=1.5)
    
    ax2.set_title('因子IC累计图', fontproperties=CHINESE_FONT, fontsize=14, fontweight='bold')
    ax2.set_xlabel('日期', fontproperties=CHINESE_FONT, fontsize=12)
    ax2.set_ylabel('累计IC', fontproperties=CHINESE_FONT, fontsize=12)
    ax2.legend(fontsize=10, loc='best', ncol=2, prop=CHINESE_FONT)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def create_dashboard(portfolio_values: pd.Series,
                    benchmark_values: pd.Series = None,
                    metrics: dict = None,
                    save_path: str = None):
    """
    创建综合看板
    
    Args:
        portfolio_values: 组合净值
        benchmark_values: 基准净值
        metrics: 绩效指标字典
        save_path: 保存路径
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. 净值曲线
    ax1 = fig.add_subplot(gs[0, :])
    portfolio_norm = portfolio_values / portfolio_values.iloc[0] * 100
    ax1.plot(portfolio_norm.index, portfolio_norm.values,
             label='策略组合', linewidth=2, color='#E74C3C')
    
    if benchmark_values is not None and len(benchmark_values) > 0:
        # 使用插值对齐基准数据
        benchmark_norm = benchmark_values / benchmark_values.iloc[0] * 100
        benchmark_aligned = benchmark_norm.reindex(
            portfolio_norm.index.union(benchmark_norm.index)
        ).interpolate(method='time').reindex(portfolio_norm.index)
        ax1.plot(benchmark_aligned.index, benchmark_aligned.values,
                 label='基准指数', linewidth=2, color='#3498DB', alpha=0.7)
    
    ax1.set_title('净值曲线', fontproperties=CHINESE_FONT, fontsize=14, fontweight='bold')
    ax1.legend(prop=CHINESE_FONT)
    ax1.grid(True, alpha=0.3)
    
    # 2. 回撤
    ax2 = fig.add_subplot(gs[1, 0])
    portfolio_norm = portfolio_values / portfolio_values.iloc[0]
    portfolio_cummax = portfolio_norm.expanding().max()
    portfolio_drawdown = (portfolio_norm - portfolio_cummax) / portfolio_cummax * 100
    ax2.fill_between(portfolio_drawdown.index, 0, portfolio_drawdown.values,
                      color='#E74C3C', alpha=0.5)
    ax2.set_title('回撤', fontproperties=CHINESE_FONT, fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. 收益率分布
    ax3 = fig.add_subplot(gs[1, 1])
    returns = portfolio_values.pct_change().dropna() * 100
    ax3.hist(returns, bins=30, color='#3498DB', alpha=0.7, edgecolor='black')
    ax3.set_title('日收益率分布', fontproperties=CHINESE_FONT, fontsize=14, fontweight='bold')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=1)
    ax3.grid(True, alpha=0.3)
    
    # 4. 月度收益
    ax4 = fig.add_subplot(gs[2, 0])
    monthly = portfolio_values.resample('M').last().pct_change() * 100
    colors = ['green' if x > 0 else 'red' for x in monthly.values]
    ax4.bar(range(len(monthly)), monthly.values, color=colors, alpha=0.7)
    ax4.set_title('月度收益', fontproperties=CHINESE_FONT, fontsize=14, fontweight='bold')
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax4.grid(True, alpha=0.3)
    
    # 5. 绩效指标
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    if metrics:
        text_lines = []
        text_lines.append("主要绩效指标")
        text_lines.append("-" * 40)
        
        key_metrics = ['total_return', 'annual_return', 'sharpe_ratio',
                      'max_drawdown', 'win_rate']
        
        metric_names = {
            'total_return': '累计收益率',
            'annual_return': '年化收益率',
            'sharpe_ratio': '夏普比率',
            'max_drawdown': '最大回撤',
            'win_rate': '胜率'
        }
        
        for key in key_metrics:
            if key in metrics:
                name = metric_names.get(key, key)
                value = metrics[key]
                if 'return' in key or 'drawdown' in key or 'rate' in key:
                    text_lines.append(f"{name}: {value:.2f}%")
                else:
                    text_lines.append(f"{name}: {value:.4f}")
        
        ax5.text(0.1, 0.9, '\n'.join(text_lines),
                fontsize=12, verticalalignment='top', fontproperties=CHINESE_FONT)
    
    plt.suptitle('策略绩效看板', fontproperties=CHINESE_FONT, fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=252, freq='D')
    returns = np.random.randn(252) * 0.02 + 0.001
    portfolio_values = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)
    
    benchmark_returns = np.random.randn(252) * 0.015 + 0.0005
    benchmark_values = pd.Series(100 * np.exp(np.cumsum(benchmark_returns)), index=dates)
    
    # 测试各种图表
    print("测试净值曲线...")
    plot_equity_curve(portfolio_values, benchmark_values)
    
    print("测试回撤曲线...")
    plot_drawdown(portfolio_values, benchmark_values)
    
    print("测试收益率分布...")
    plot_returns_distribution(portfolio_values, benchmark_values)
