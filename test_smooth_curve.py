"""
测试平滑曲线修复
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from src.utils.visualization_fixed import plot_equity_curve, plot_drawdown

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 生成测试数据
print("生成测试数据...")

# 生成完整的交易日序列（每周5天）
start_date = datetime(2020, 1, 1)
end_date = datetime(2025, 11, 4)
all_dates = pd.date_range(start=start_date, end=end_date, freq='B')  # 工作日

# 组合净值：每20天调仓一次（模拟月度调仓）
rebalance_dates = all_dates[::20]  # 每20个交易日
portfolio_values = pd.Series(
    100 * np.exp(np.cumsum(np.random.randn(len(rebalance_dates)) * 0.02)),
    index=rebalance_dates
)

# 基准净值：每个交易日都有数据
benchmark_values = pd.Series(
    100 * np.exp(np.cumsum(np.random.randn(len(all_dates)) * 0.015)),
    index=all_dates
)

print(f"组合数据点数: {len(portfolio_values)}")
print(f"基准数据点数: {len(benchmark_values)}")
print(f"组合日期范围: {portfolio_values.index[0]} 至 {portfolio_values.index[-1]}")
print(f"基准日期范围: {benchmark_values.index[0]} 至 {benchmark_values.index[-1]}")

# 测试原始方法（ffill）- 会产生横跳
print("\n测试方法1: 使用ffill填充（旧方法）...")
benchmark_ffill = benchmark_values.reindex(portfolio_values.index).fillna(method='ffill')
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(portfolio_values.index, portfolio_values / portfolio_values.iloc[0] * 100,
        label='策略组合', linewidth=2, color='#E74C3C')
ax.plot(benchmark_ffill.index, benchmark_ffill / benchmark_ffill.iloc[0] * 100,
        label='基准指数 (ffill)', linewidth=2, color='#3498DB', alpha=0.7)
ax.set_title('旧方法：使用ffill（会产生横跳）', fontsize=16, fontweight='bold')
ax.set_xlabel('日期', fontsize=12)
ax.set_ylabel('净值（归一化）', fontsize=12)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/test_curve_ffill.png', dpi=300, bbox_inches='tight')
print("  已保存: results/figures/test_curve_ffill.png")
plt.close()

# 测试新方法1：只使用共同日期（不填充）
print("\n测试方法2: 只使用共同日期（无填充）...")
common_dates = portfolio_values.index.intersection(benchmark_values.index)
portfolio_common = portfolio_values[common_dates]
benchmark_common = benchmark_values[common_dates]

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(portfolio_values.index, portfolio_values / portfolio_values.iloc[0] * 100,
        label='策略组合', linewidth=2, color='#E74C3C')
ax.plot(benchmark_common.index, benchmark_common / benchmark_common.iloc[0] * 100,
        label='基准指数 (共同日期)', linewidth=2, color='#3498DB', alpha=0.7)
ax.set_title('新方法1：只使用共同日期', fontsize=16, fontweight='bold')
ax.set_xlabel('日期', fontsize=12)
ax.set_ylabel('净值（归一化）', fontsize=12)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/test_curve_common.png', dpi=300, bbox_inches='tight')
print("  已保存: results/figures/test_curve_common.png")
plt.close()

# 测试新方法2：线性插值
print("\n测试方法3: 使用线性插值...")
benchmark_interp = benchmark_values.reindex(
    portfolio_values.index.union(benchmark_values.index)
).interpolate(method='time').reindex(portfolio_values.index)

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(portfolio_values.index, portfolio_values / portfolio_values.iloc[0] * 100,
        label='策略组合', linewidth=2, color='#E74C3C')
ax.plot(benchmark_interp.index, benchmark_interp / benchmark_interp.iloc[0] * 100,
        label='基准指数 (插值)', linewidth=2, color='#3498DB', alpha=0.7)
ax.set_title('新方法2：使用线性插值（最平滑）', fontsize=16, fontweight='bold')
ax.set_xlabel('日期', fontsize=12)
ax.set_ylabel('净值（归一化）', fontsize=12)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/test_curve_interpolate.png', dpi=300, bbox_inches='tight')
print("  已保存: results/figures/test_curve_interpolate.png")
plt.close()

# 测试优化后的绘图函数
print("\n测试方法4: 使用优化后的绘图函数...")
plot_equity_curve(
    portfolio_values=portfolio_values,
    benchmark_values=benchmark_values,
    title='优化后的净值曲线（使用插值）',
    save_path='results/figures/test_curve_optimized.png'
)

plot_drawdown(
    portfolio_values=portfolio_values,
    benchmark_values=benchmark_values,
    title='优化后的回撤曲线（使用插值）',
    save_path='results/figures/test_drawdown_optimized.png'
)

print("\n测试完成！请查看以下文件对比效果：")
print("  1. results/figures/test_curve_ffill.png - 旧方法（横跳）")
print("  2. results/figures/test_curve_common.png - 只用共同日期")
print("  3. results/figures/test_curve_interpolate.png - 线性插值（推荐）")
print("  4. results/figures/test_curve_optimized.png - 优化后的函数")
print("  5. results/figures/test_drawdown_optimized.png - 优化后的回撤")
