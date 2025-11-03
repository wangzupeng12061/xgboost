"""
组合管理模块
负责持仓管理、调仓、交易成本计算
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class PortfolioManager:
    """组合管理类"""
    
    def __init__(self,
                 initial_capital: float = 1000000,
                 commission_rate: float = 0.0003,
                 min_commission: float = 5,
                 slippage: float = 0.001):
        """
        初始化组合管理器
        
        Args:
            initial_capital: 初始资金
            commission_rate: 佣金费率（双边）
            min_commission: 最低佣金
            slippage: 滑点（买入加价、卖出减价）
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.slippage = slippage
        
        # 持仓信息: {stock_code: {'shares': xxx, 'cost': xxx, 'value': xxx}}
        self.positions = {}
        
        # 现金
        self.cash = initial_capital
        
        # 组合市值
        self.portfolio_value = initial_capital
        
        # 交易记录
        self.trades = []
        
        # 每日持仓记录
        self.daily_positions = []
        
        # 每日净值记录
        self.daily_values = []
        
        print(f"PortfolioManager initialized with capital: {initial_capital:,.0f}")
    
    def rebalance(self,
                 date: str,
                 target_positions: pd.DataFrame,
                 current_prices: Dict[str, float]) -> Dict:
        """
        调仓
        
        Args:
            date: 日期
            target_positions: 目标持仓 (包含stock_code和weight列)
            current_prices: 当前价格字典 {stock_code: price}
            
        Returns:
            调仓结果字典
        """
        print(f"\n[{date}] 开始调仓...")
        
        # 更新组合市值
        self._update_portfolio_value(current_prices)
        
        # 计算目标持仓金额
        target_dict = {}
        for _, row in target_positions.iterrows():
            stock_code = row['stock_code']
            weight = row['weight']
            target_value = self.portfolio_value * weight
            target_dict[stock_code] = target_value
        
        # 卖出不在目标中的股票
        stocks_to_sell = set(self.positions.keys()) - set(target_dict.keys())
        for stock_code in stocks_to_sell:
            if stock_code in current_prices:
                shares = self.positions[stock_code]['shares']
                self._sell_stock(date, stock_code, shares, current_prices[stock_code])
        
        # 调整持仓
        for stock_code, target_value in target_dict.items():
            if stock_code not in current_prices:
                print(f"  警告: {stock_code} 价格缺失，跳过")
                continue
            
            current_value = self._get_position_value(stock_code, current_prices)
            price = current_prices[stock_code]
            
            if current_value < target_value * 0.95:  # 需要买入（允许5%误差）
                buy_value = target_value - current_value
                shares = int(buy_value / price / 100) * 100  # 整手
                if shares >= 100:  # 至少1手
                    self._buy_stock(date, stock_code, shares, price)
                    
            elif current_value > target_value * 1.05:  # 需要卖出
                sell_value = current_value - target_value
                shares = int(sell_value / price / 100) * 100
                if shares >= 100:
                    self._sell_stock(date, stock_code, shares, price)
        
        # 更新市值
        self._update_portfolio_value(current_prices)
        
        # 记录每日持仓
        self._record_daily_position(date, current_prices)
        
        result = {
            'date': date,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'cash_ratio': self.cash / self.portfolio_value,
            'n_positions': len(self.positions),
            'position_value': self.portfolio_value - self.cash
        }
        
        print(f"  组合市值: {self.portfolio_value:,.0f}")
        print(f"  现金: {self.cash:,.0f} ({result['cash_ratio']:.1%})")
        print(f"  持仓数: {len(self.positions)}")
        
        return result
    
    def _buy_stock(self,
                  date: str,
                  stock_code: str,
                  shares: int,
                  price: float):
        """
        买入股票
        
        Args:
            date: 日期
            stock_code: 股票代码
            shares: 股数
            price: 价格
        """
        # 考虑滑点（买入加价）
        execution_price = price * (1 + self.slippage)
        
        # 计算成本
        cost = shares * execution_price
        commission = max(cost * self.commission_rate, self.min_commission)
        total_cost = cost + commission
        
        # 检查资金是否足够
        if total_cost > self.cash:
            # 调整买入数量
            available_cash = self.cash - self.min_commission
            shares = int(available_cash / (execution_price * (1 + self.commission_rate)) / 100) * 100
            
            if shares < 100:  # 资金不足买1手
                return
            
            cost = shares * execution_price
            commission = max(cost * self.commission_rate, self.min_commission)
            total_cost = cost + commission
        
        # 更新持仓
        if stock_code in self.positions:
            old_shares = self.positions[stock_code]['shares']
            old_cost = self.positions[stock_code]['cost']
            new_shares = old_shares + shares
            new_cost = (old_cost * old_shares + execution_price * shares) / new_shares
            
            self.positions[stock_code] = {
                'shares': new_shares,
                'cost': new_cost,
                'value': new_shares * execution_price
            }
        else:
            self.positions[stock_code] = {
                'shares': shares,
                'cost': execution_price,
                'value': shares * execution_price
            }
        
        # 扣除现金
        self.cash -= total_cost
        
        # 记录交易
        self.trades.append({
            'date': date,
            'stock_code': stock_code,
            'action': 'buy',
            'shares': shares,
            'price': price,
            'execution_price': execution_price,
            'commission': commission,
            'total_cost': total_cost
        })
    
    def _sell_stock(self,
                   date: str,
                   stock_code: str,
                   shares: int,
                   price: float):
        """
        卖出股票
        
        Args:
            date: 日期
            stock_code: 股票代码
            shares: 股数
            price: 价格
        """
        if stock_code not in self.positions:
            return
        
        # 调整卖出数量（不能超过持仓）
        max_shares = self.positions[stock_code]['shares']
        shares = min(shares, max_shares)
        
        if shares < 100:  # 不足1手，全部卖出
            shares = max_shares
        
        # 考虑滑点（卖出减价）
        execution_price = price * (1 - self.slippage)
        
        # 计算收入
        revenue = shares * execution_price
        commission = max(revenue * self.commission_rate, self.min_commission)
        stamp_tax = revenue * 0.001  # 印花税（单边）
        total_cost = commission + stamp_tax
        net_revenue = revenue - total_cost
        
        # 更新持仓
        remaining_shares = max_shares - shares
        if remaining_shares > 0:
            self.positions[stock_code]['shares'] = remaining_shares
            self.positions[stock_code]['value'] = remaining_shares * execution_price
        else:
            del self.positions[stock_code]
        
        # 增加现金
        self.cash += net_revenue
        
        # 记录交易
        self.trades.append({
            'date': date,
            'stock_code': stock_code,
            'action': 'sell',
            'shares': shares,
            'price': price,
            'execution_price': execution_price,
            'commission': commission,
            'stamp_tax': stamp_tax,
            'total_cost': total_cost,
            'net_revenue': net_revenue
        })
    
    def _update_portfolio_value(self, current_prices: Dict[str, float]):
        """更新组合市值"""
        position_value = 0
        
        for stock_code, position in self.positions.items():
            if stock_code in current_prices:
                current_price = current_prices[stock_code]
                value = position['shares'] * current_price
                position['value'] = value
                position_value += value
            else:
                # 价格缺失，使用成本价
                position_value += position['value']
        
        self.portfolio_value = self.cash + position_value
    
    def _get_position_value(self,
                           stock_code: str,
                           current_prices: Dict[str, float]) -> float:
        """获取持仓市值"""
        if stock_code not in self.positions:
            return 0
        
        shares = self.positions[stock_code]['shares']
        price = current_prices.get(stock_code, self.positions[stock_code]['cost'])
        
        return shares * price
    
    def _record_daily_position(self, date: str, current_prices: Dict[str, float]):
        """记录每日持仓"""
        positions_snapshot = []
        
        for stock_code, position in self.positions.items():
            price = current_prices.get(stock_code, position['cost'])
            value = position['shares'] * price
            
            positions_snapshot.append({
                'date': date,
                'stock_code': stock_code,
                'shares': position['shares'],
                'cost': position['cost'],
                'price': price,
                'value': value,
                'profit': value - position['shares'] * position['cost'],
                'profit_pct': (price / position['cost'] - 1) * 100
            })
        
        self.daily_positions.extend(positions_snapshot)
        
        # 记录每日净值
        self.daily_values.append({
            'date': date,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'position_value': self.portfolio_value - self.cash,
            'n_positions': len(self.positions),
            'return': (self.portfolio_value / self.initial_capital - 1) * 100
        })
    
    def get_trades_df(self) -> pd.DataFrame:
        """获取交易记录DataFrame"""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)
    
    def get_daily_positions_df(self) -> pd.DataFrame:
        """获取每日持仓记录"""
        if not self.daily_positions:
            return pd.DataFrame()
        return pd.DataFrame(self.daily_positions)
    
    def get_daily_values_df(self) -> pd.DataFrame:
        """获取每日净值记录"""
        if not self.daily_values:
            return pd.DataFrame()
        return pd.DataFrame(self.daily_values)
    
    def get_current_positions(self) -> pd.DataFrame:
        """获取当前持仓"""
        if not self.positions:
            return pd.DataFrame()
        
        positions_list = []
        for stock_code, position in self.positions.items():
            positions_list.append({
                'stock_code': stock_code,
                'shares': position['shares'],
                'cost': position['cost'],
                'value': position['value'],
                'weight': position['value'] / self.portfolio_value
            })
        
        return pd.DataFrame(positions_list)
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        trades_df = self.get_trades_df()
        
        if len(trades_df) == 0:
            return {}
        
        buy_trades = trades_df[trades_df['action'] == 'buy']
        sell_trades = trades_df[trades_df['action'] == 'sell']
        
        stats = {
            'total_trades': len(trades_df),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'total_commission': trades_df['commission'].sum(),
            'total_stamp_tax': sell_trades.get('stamp_tax', pd.Series([0])).sum(),
            'total_cost': trades_df['total_cost'].sum(),
            'turnover_value': buy_trades['total_cost'].sum(),
            'current_value': self.portfolio_value,
            'total_return': (self.portfolio_value / self.initial_capital - 1) * 100,
            'cash_ratio': self.cash / self.portfolio_value * 100
        }
        
        return stats
    
    def reset(self):
        """重置组合"""
        self.capital = self.initial_capital
        self.cash = self.initial_capital
        self.portfolio_value = self.initial_capital
        self.positions = {}
        self.trades = []
        self.daily_positions = []
        self.daily_values = []
        print("Portfolio reset")


def test_portfolio_manager():
    """测试组合管理功能"""
    print("="*60)
    print("测试组合管理")
    print("="*60)
    
    # 初始化组合管理器
    pm = PortfolioManager(initial_capital=1000000)
    
    # 模拟第一天：买入3只股票
    print("\n第1天: 初始建仓")
    target_positions = pd.DataFrame({
        'stock_code': ['000001', '000002', '000003'],
        'weight': [0.3, 0.3, 0.3]
    })
    
    current_prices = {
        '000001': 10.0,
        '000002': 20.0,
        '000003': 15.0
    }
    
    result1 = pm.rebalance('2024-01-01', target_positions, current_prices)
    
    # 查看持仓
    print("\n当前持仓:")
    print(pm.get_current_positions())
    
    # 模拟第二天：价格变化，调仓
    print("\n第2天: 调仓")
    target_positions = pd.DataFrame({
        'stock_code': ['000001', '000002', '000004'],  # 换股
        'weight': [0.4, 0.3, 0.2]
    })
    
    current_prices = {
        '000001': 11.0,  # 上涨10%
        '000002': 19.0,  # 下跌5%
        '000003': 14.5,  # 下跌3.3%
        '000004': 25.0
    }
    
    result2 = pm.rebalance('2024-01-02', target_positions, current_prices)
    
    # 查看持仓
    print("\n当前持仓:")
    print(pm.get_current_positions())
    
    # 查看交易记录
    print("\n交易记录:")
    print(pm.get_trades_df())
    
    # 查看统计
    print("\n统计信息:")
    stats = pm.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:,.2f}")
        else:
            print(f"  {key}: {value}")
    
    # 查看每日净值
    print("\n每日净值:")
    print(pm.get_daily_values_df())


if __name__ == "__main__":
    test_portfolio_manager()
