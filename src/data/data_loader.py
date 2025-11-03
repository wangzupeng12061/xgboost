"""
数据加载模块
支持从Tushare、AKShare等数据源加载股票数据
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False
    print("Warning: tushare not installed")

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    print("Warning: akshare not installed")


class DataLoader:
    """数据加载类"""
    
    def __init__(self, source: str = 'tushare', token: str = None):
        """
        初始化数据加载器
        
        Args:
            source: 数据源 ('tushare', 'akshare', 'local')
            token: tushare token
        """
        self.source = source
        
        if source == 'tushare':
            if not TUSHARE_AVAILABLE:
                raise ImportError("tushare not installed. Run: pip install tushare")
            if token:
                ts.set_token(token)
                self.pro = ts.pro_api()
            else:
                raise ValueError("Tushare token is required")
        
        elif source == 'akshare':
            if not AKSHARE_AVAILABLE:
                raise ImportError("akshare not installed. Run: pip install akshare")
        
        print(f"DataLoader initialized with source: {source}")
    
    def load_stock_list(self, 
                       date: str = None,
                       market: str = 'all') -> pd.DataFrame:
        """
        加载股票列表
        
        Args:
            date: 交易日期 (YYYYMMDD)
            market: 市场类型 ('all', 'main', 'gem', 'star')
            
        Returns:
            股票列表DataFrame
        """
        if self.source == 'tushare':
            return self._load_stock_list_tushare(date, market)
        elif self.source == 'akshare':
            return self._load_stock_list_akshare()
        else:
            raise ValueError(f"Unsupported source: {self.source}")
    
    def _load_stock_list_tushare(self, date: str = None, market: str = 'all') -> pd.DataFrame:
        """从Tushare加载股票列表"""
        # 获取所有上市股票
        stock_list = self.pro.stock_basic(
            exchange='',
            list_status='L',
            fields='ts_code,symbol,name,area,industry,market,list_date'
        )
        
        # 筛选市场
        if market != 'all':
            market_map = {
                'main': '主板',
                'gem': '创业板',
                'star': '科创板'
            }
            if market in market_map:
                stock_list = stock_list[stock_list['market'] == market_map[market]]
        
        return stock_list
    
    def _load_stock_list_akshare(self) -> pd.DataFrame:
        """从AKShare加载股票列表"""
        stock_list = ak.stock_info_a_code_name()
        stock_list.columns = ['ts_code', 'name']
        return stock_list
    
    def load_daily_data(self,
                       start_date: str,
                       end_date: str,
                       stock_codes: List[str] = None) -> pd.DataFrame:
        """
        加载日线行情数据
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            stock_codes: 股票代码列表（None表示全部）
            
        Returns:
            日线数据DataFrame
        """
        if self.source == 'tushare':
            return self._load_daily_tushare(start_date, end_date, stock_codes)
        elif self.source == 'akshare':
            return self._load_daily_akshare(start_date, end_date, stock_codes)
        else:
            raise ValueError(f"Unsupported source: {self.source}")
    
    def _load_daily_tushare(self, start_date: str, end_date: str, 
                           stock_codes: List[str] = None) -> pd.DataFrame:
        """从Tushare加载日线数据"""
        start_date = start_date.replace('-', '')
        end_date = end_date.replace('-', '')
        
        if stock_codes is None:
            # 加载所有股票
            df = self.pro.daily(
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,trade_date,open,high,low,close,vol,amount,pct_chg'
            )
        else:
            # 加载指定股票
            dfs = []
            for code in stock_codes:
                df_code = self.pro.daily(
                    ts_code=code,
                    start_date=start_date,
                    end_date=end_date,
                    fields='ts_code,trade_date,open,high,low,close,vol,amount,pct_chg'
                )
                dfs.append(df_code)
            df = pd.concat(dfs, ignore_index=True)
        
        # 重命名列
        df = df.rename(columns={
            'ts_code': 'stock_code',
            'trade_date': 'date',
            'vol': 'volume'
        })
        
        # 转换日期格式
        df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    def _load_daily_akshare(self, start_date: str, end_date: str,
                           stock_codes: List[str] = None) -> pd.DataFrame:
        """从AKShare加载日线数据"""
        # AKShare需要逐个股票加载
        if stock_codes is None:
            stock_list = self.load_stock_list()
            stock_codes = stock_list['ts_code'].tolist()[:100]  # 限制数量避免过慢
        
        dfs = []
        for code in stock_codes:
            try:
                df_code = ak.stock_zh_a_hist(
                    symbol=code,
                    start_date=start_date.replace('-', ''),
                    end_date=end_date.replace('-', ''),
                    adjust="qfq"
                )
                df_code['stock_code'] = code
                dfs.append(df_code)
            except Exception as e:
                print(f"Error loading {code}: {e}")
                continue
        
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            return df
        else:
            return pd.DataFrame()
    
    def load_financial_data(self,
                           start_date: str,
                           end_date: str,
                           report_type: str = 'all') -> pd.DataFrame:
        """
        加载财务数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            report_type: 报表类型 ('balance', 'income', 'cashflow', 'all')
            
        Returns:
            财务数据DataFrame
        """
        if self.source == 'tushare':
            return self._load_financial_tushare(start_date, end_date, report_type)
        else:
            raise NotImplementedError(f"Financial data not implemented for {self.source}")
    
    def _load_financial_tushare(self, start_date: str, end_date: str,
                               report_type: str = 'all') -> pd.DataFrame:
        """从Tushare加载财务数据"""
        start_date = start_date.replace('-', '')
        end_date = end_date.replace('-', '')
        
        dfs = []
        
        # 资产负债表
        if report_type in ['balance', 'all']:
            balance = self.pro.balancesheet(
                period=start_date,
                fields='ts_code,end_date,total_assets,total_liab,total_equity'
            )
            dfs.append(balance)
        
        # 利润表
        if report_type in ['income', 'all']:
            income = self.pro.income(
                period=start_date,
                fields='ts_code,end_date,revenue,operate_profit,total_profit,n_income'
            )
            dfs.append(income)
        
        # 现金流量表
        if report_type in ['cashflow', 'all']:
            cashflow = self.pro.cashflow(
                period=start_date,
                fields='ts_code,end_date,n_cashflow_act,n_cashflow_inv,n_cash_flows'
            )
            dfs.append(cashflow)
        
        # 合并数据
        if dfs:
            df = dfs[0]
            for i in range(1, len(dfs)):
                df = pd.merge(df, dfs[i], on=['ts_code', 'end_date'], how='outer')
            
            df = df.rename(columns={'ts_code': 'stock_code', 'end_date': 'report_date'})
            df['report_date'] = pd.to_datetime(df['report_date'])
            
            return df
        else:
            return pd.DataFrame()
    
    def load_index_data(self,
                       index_code: str,
                       start_date: str,
                       end_date: str) -> pd.DataFrame:
        """
        加载指数数据
        
        Args:
            index_code: 指数代码（如 '000300.SH' 沪深300）
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            指数数据DataFrame
        """
        if self.source == 'tushare':
            return self._load_index_tushare(index_code, start_date, end_date)
        elif self.source == 'akshare':
            return self._load_index_akshare(index_code, start_date, end_date)
        else:
            raise ValueError(f"Unsupported source: {self.source}")
    
    def _load_index_tushare(self, index_code: str, start_date: str, 
                           end_date: str) -> pd.DataFrame:
        """从Tushare加载指数数据"""
        start_date = start_date.replace('-', '')
        end_date = end_date.replace('-', '')
        
        df = self.pro.index_daily(
            ts_code=index_code,
            start_date=start_date,
            end_date=end_date,
            fields='ts_code,trade_date,close,pct_chg'
        )
        
        df = df.rename(columns={'trade_date': 'date', 'pct_chg': 'return'})
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        return df
    
    def _load_index_akshare(self, index_code: str, start_date: str,
                           end_date: str) -> pd.DataFrame:
        """从AKShare加载指数数据"""
        # 转换指数代码格式
        index_map = {
            '000300.SH': 'sh000300',
            '000905.SH': 'sh000905',
            '000852.SH': 'sh000852'
        }
        
        ak_code = index_map.get(index_code, index_code)
        
        df = ak.stock_zh_index_daily(symbol=ak_code)
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        return df
    
    def load_market_data(self, date: str) -> Dict:
        """
        加载市场数据（流通市值、PE等）
        
        Args:
            date: 日期
            
        Returns:
            市场数据字典
        """
        if self.source == 'tushare':
            date = date.replace('-', '')
            df = self.pro.daily_basic(
                trade_date=date,
                fields='ts_code,trade_date,turnover_rate,pe,pb,ps,total_mv,circ_mv'
            )
            
            df = df.rename(columns={
                'ts_code': 'stock_code',
                'trade_date': 'date',
                'total_mv': 'market_cap',
                'circ_mv': 'float_market_cap'
            })
            
            df['date'] = pd.to_datetime(df['date'])
            
            return df
        else:
            raise NotImplementedError(f"Market data not implemented for {self.source}")


def test_data_loader():
    """测试数据加载功能"""
    # 使用AKShare测试（无需token）
    loader = DataLoader(source='akshare')
    
    # 测试加载股票列表
    print("Testing load_stock_list...")
    stock_list = loader.load_stock_list()
    print(f"Loaded {len(stock_list)} stocks")
    print(stock_list.head())
    
    # 测试加载日线数据（限制数量）
    print("\nTesting load_daily_data...")
    daily_data = loader.load_daily_data(
        start_date='2024-01-01',
        end_date='2024-01-31',
        stock_codes=['000001', '000002']
    )
    print(f"Loaded {len(daily_data)} daily records")
    print(daily_data.head())


if __name__ == "__main__":
    test_data_loader()
