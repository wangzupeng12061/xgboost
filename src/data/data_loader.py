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
        try:
            # 获取所有上市股票（使用日线数据接口，不需要高级权限）
            if date is None:
                date = datetime.now().strftime('%Y%m%d')
            else:
                date = date.replace('-', '')
            
            # 使用交易日历获取最近的交易日
            cal = self.pro.trade_cal(exchange='SSE', start_date=date, end_date=date, is_open='1')
            if len(cal) == 0:
                # 如果当天不是交易日，获取最近的交易日
                cal = self.pro.trade_cal(exchange='SSE', end_date=date, is_open='1')
                if len(cal) > 0:
                    date = cal.iloc[-1]['cal_date']
            
            # 通过日线数据获取股票列表
            df_daily = self.pro.daily(trade_date=date, fields='ts_code')
            stock_codes = df_daily['ts_code'].unique().tolist()
            
            print(f"从交易日 {date} 获取到 {len(stock_codes)} 只股票")
            
            # 创建股票列表DataFrame
            stock_list = pd.DataFrame({
                'ts_code': stock_codes,
                'stock_code': stock_codes
            })
            
            return stock_list
            
        except Exception as e:
            print(f"加载股票列表失败: {e}")
            # 返回一些常见股票作为fallback
            fallback_stocks = [
                '000001.SZ', '000002.SZ', '600000.SH', '600519.SH', 
                '000858.SZ', '601318.SH', '600036.SH', '000333.SZ'
            ]
            print(f"使用备用股票列表: {len(fallback_stocks)} 只股票")
            return pd.DataFrame({
                'ts_code': fallback_stocks,
                'stock_code': fallback_stocks
            })
    
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
        
        print(f"开始加载日线数据: {start_date} 至 {end_date}")
        
        # 强制使用逐股票加载方式，避免API频率限制
        if stock_codes is None or len(stock_codes) > 200:
            # 只有股票数量极多时才按日期批量加载
            print(f"按日期批量加载数据（股票数量较多）...")
            
            # 获取交易日历
            cal = self.pro.trade_cal(
                exchange='SSE',
                start_date=start_date,
                end_date=end_date,
                is_open='1'
            )
            trade_dates = cal['cal_date'].tolist()
            
            dfs = []
            for i, date in enumerate(trade_dates):
                try:
                    df_date = self.pro.daily(
                        trade_date=date,
                        fields='ts_code,trade_date,open,high,low,close,vol,amount,pct_chg'
                    )
                    
                    if len(df_date) > 0:
                        # 如果指定了股票列表，进行筛选
                        if stock_codes is not None:
                            df_date = df_date[df_date['ts_code'].isin(stock_codes)]
                        dfs.append(df_date)
                    
                    if (i + 1) % 10 == 0:
                        print(f"  已加载 {i+1}/{len(trade_dates)} 个交易日")
                    
                    # API限流，稍微延迟
                    import time
                    time.sleep(0.5)  # 增加等待时间避免频率限制
                    
                except Exception as e:
                    error_msg = str(e)
                    if "每分钟最多访问" in error_msg or "访问频率" in error_msg:
                        print(f"  触发API频率限制，等待60秒...")
                        import time
                        time.sleep(60)
                    print(f"  加载日期 {date} 失败: {e}")
                    continue
            
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
                print(f"成功加载 {len(df)} 条记录")
            else:
                print("未加载到任何数据")
                df = pd.DataFrame()
        else:
            # 股票数量较少，逐个加载
            print(f"加载 {len(stock_codes)} 只股票的数据...")
            dfs = []
            for i, code in enumerate(stock_codes):
                try:
                    df_code = self.pro.daily(
                        ts_code=code,
                        start_date=start_date,
                        end_date=end_date,
                        fields='ts_code,trade_date,open,high,low,close,vol,amount,pct_chg'
                    )
                    if len(df_code) > 0:
                        dfs.append(df_code)
                    
                    if (i + 1) % 10 == 0:
                        print(f"  已加载 {i+1}/{len(stock_codes)} 只股票")
                    
                    # API限流
                    import time
                    time.sleep(0.5)  # 增加等待时间避免频率限制
                    
                except Exception as e:
                    error_msg = str(e)
                    if "每分钟最多访问" in error_msg or "访问频率" in error_msg:
                        print(f"  触发API频率限制，等待60秒...")
                        import time
                        time.sleep(60)
                    print(f"  加载股票 {code} 失败: {e}")
                    continue
            
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
                print(f"成功加载 {len(df)} 条记录")
            else:
                df = pd.DataFrame()
        
        if len(df) > 0:
            # 重命名列
            df = df.rename(columns={
                'ts_code': 'stock_code',
                'trade_date': 'date',
                'vol': 'volume'
            })
            
            # 转换日期格式
            df['date'] = pd.to_datetime(df['date'])
            
            # 排序
            df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
        
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
    
    def load_industry_data(self, stock_codes: List[str] = None) -> pd.DataFrame:
        """
        加载行业分类数据
        
        Args:
            stock_codes: 股票代码列表，如果为None则加载全部
            
        Returns:
            行业分类DataFrame，包含stock_code, industry, industry_code等字段
        """
        if self.source == 'tushare':
            # 获取股票行业分类（申万行业）
            if stock_codes is None:
                # 获取所有股票
                df = self.pro.stock_basic(
                    exchange='',
                    list_status='L',
                    fields='ts_code,name,industry'
                )
            else:
                # 分批获取指定股票
                dfs = []
                batch_size = 100
                for i in range(0, len(stock_codes), batch_size):
                    batch = stock_codes[i:i+batch_size]
                    # tushare不支持直接按股票列表查询，需要逐个查询或使用stock_basic
                    for code in batch:
                        try:
                            df_code = self.pro.stock_basic(
                                ts_code=code,
                                fields='ts_code,name,industry'
                            )
                            if len(df_code) > 0:
                                dfs.append(df_code)
                        except Exception as e:
                            print(f"  加载股票 {code} 行业数据失败: {e}")
                            continue
                
                if dfs:
                    df = pd.concat(dfs, ignore_index=True)
                else:
                    df = pd.DataFrame()
            
            if len(df) > 0:
                df = df.rename(columns={'ts_code': 'stock_code'})
                
                # 添加行业编码（简单映射）
                df['industry_code'] = df['industry'].factorize()[0]
                
            return df
        else:
            raise NotImplementedError(f"Industry data not implemented for {self.source}")
    
    def load_financial_indicators(self, 
                                   start_date: str,
                                   end_date: str,
                                   stock_codes: List[str] = None,
                                   indicators: List[str] = None) -> pd.DataFrame:
        """
        加载财务指标数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            stock_codes: 股票代码列表
            indicators: 指标列表，默认加载常用指标
            
        Returns:
            财务指标DataFrame
        """
        if self.source == 'tushare':
            start_date = start_date.replace('-', '')
            end_date = end_date.replace('-', '')
            
            # 默认指标：ROE, ROA, 营业收入, 净利润, 资产负债率等
            if indicators is None:
                fields = 'ts_code,end_date,roe,roa,debt_to_assets,revenue_yoy,net_profit_yoy,gross_profit_margin,net_profit_margin'
            else:
                fields = 'ts_code,end_date,' + ','.join(indicators)
            
            dfs = []
            
            # 如果指定了股票代码，逐个加载
            if stock_codes is not None:
                print(f"加载 {len(stock_codes)} 只股票的财务指标...")
                for i, code in enumerate(stock_codes):
                    try:
                        df_code = self.pro.fina_indicator(
                            ts_code=code,
                            start_date=start_date,
                            end_date=end_date,
                            fields=fields
                        )
                        if len(df_code) > 0:
                            dfs.append(df_code)
                        
                        if (i + 1) % 10 == 0:
                            print(f"  已加载 {i+1}/{len(stock_codes)} 只股票")
                        
                        import time
                        time.sleep(0.5)  # API限流，增加等待时间
                    except Exception as e:
                        error_msg = str(e)
                        if "每分钟最多访问" in error_msg or "访问频率" in error_msg:
                            print(f"  触发API频率限制，等待60秒...")
                            import time
                            time.sleep(60)
                        print(f"  加载股票 {code} 财务指标失败: {e}")
                        continue
            else:
                # 按日期范围加载所有股票
                try:
                    df = self.pro.fina_indicator(
                        start_date=start_date,
                        end_date=end_date,
                        fields=fields
                    )
                    dfs.append(df)
                except Exception as e:
                    print(f"加载财务指标失败: {e}")
            
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
                df = df.rename(columns={'ts_code': 'stock_code'})
                df['end_date'] = pd.to_datetime(df['end_date'])
                
                # 去除重复行（Tushare可能返回重复数据）
                df = df.drop_duplicates(subset=['stock_code', 'end_date'], keep='first')
                print(f"✓ 去重后保留 {len(df)} 条财务记录")
                
                return df
            else:
                return pd.DataFrame()
        else:
            raise NotImplementedError(f"Financial indicators not implemented for {self.source}")
    
    def load_macro_data(self, 
                       start_date: str,
                       end_date: str) -> pd.DataFrame:
        """
        加载宏观经济数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            宏观数据DataFrame，包含CPI、PPI、PMI、M2、利率等指标
        """
        if self.source == 'tushare':
            start_date = start_date.replace('-', '')
            end_date = end_date.replace('-', '')
            
            macro_dfs = []
            
            # 1. CPI和PPI数据（月度）
            try:
                print("  加载CPI/PPI数据...")
                cpi_ppi = self.pro.cn_cpi(
                    start_date=start_date[:6],  # 月度数据，只需YYYYMM
                    end_date=end_date[:6],
                    fields='month,nt_yoy,nt_mom,nt_val'  # 全国CPI同比、环比、值
                )
                if len(cpi_ppi) > 0:
                    cpi_ppi['date'] = pd.to_datetime(cpi_ppi['month'], format='%Y%m')
                    cpi_ppi = cpi_ppi.rename(columns={
                        'nt_yoy': 'cpi_yoy',
                        'nt_mom': 'cpi_mom',
                        'nt_val': 'cpi_val'
                    })
                    cpi_ppi = cpi_ppi[['date', 'cpi_yoy', 'cpi_mom', 'cpi_val']]
                    macro_dfs.append(cpi_ppi)
                    print(f"    ✓ CPI数据: {len(cpi_ppi)}条")
            except Exception as e:
                print(f"    加载CPI数据失败: {e}")
            
            # 2. PMI数据（月度）
            try:
                print("  加载PMI数据...")
                pmi = self.pro.cn_pmi(
                    start_date=start_date[:6],
                    end_date=end_date[:6]
                )
                if len(pmi) > 0:
                    # PMI数据返回的是详细指标，需要找到正确的列
                    # MONTH列是日期，PMI010000是制造业PMI指数
                    if 'MONTH' in pmi.columns:
                        pmi['date'] = pd.to_datetime(pmi['MONTH'], format='%Y%m')
                        
                        # PMI010000 是制造业PMI指数
                        pmi_cols = ['date']
                        if 'PMI010000' in pmi.columns:
                            pmi = pmi.rename(columns={'PMI010000': 'pmi'})
                            pmi_cols.append('pmi')
                        
                        pmi = pmi[pmi_cols]
                        
                        # 合并PMI数据
                        if len(macro_dfs) > 0:
                            macro_dfs[0] = pd.merge(macro_dfs[0], pmi, on='date', how='outer')
                        else:
                            macro_dfs.append(pmi)
                        print(f"    ✓ PMI数据: {len(pmi)}条")
                    else:
                        print("    ✗ PMI数据格式不符合预期")
            except Exception as e:
                print(f"    加载PMI数据失败: {e}")


            
            # 3. M2货币供应量（月度）
            try:
                print("  加载M2数据...")
                m2 = self.pro.cn_m(
                    start_date=start_date[:6],
                    end_date=end_date[:6],
                    fields='month,m2,m2_yoy'
                )
                if len(m2) > 0:
                    m2['date'] = pd.to_datetime(m2['month'], format='%Y%m')
                    m2 = m2[['date', 'm2', 'm2_yoy']]
                    
                    # 合并M2数据
                    if len(macro_dfs) > 0:
                        macro_dfs[0] = pd.merge(macro_dfs[0], m2, on='date', how='outer')
                    else:
                        macro_dfs.append(m2)
                    print(f"    ✓ M2数据: {len(m2)}条")
            except Exception as e:
                print(f"    加载M2数据失败: {e}")
            
            # 4. 利率数据（按需采样）
            try:
                print("  加载利率数据...")
                # 使用Shibor隔夜利率作为短期利率代理
                shibor = self.pro.shibor(
                    start_date=start_date,
                    end_date=end_date,
                    fields='date,on,1w,1m'  # 隔夜、1周、1月利率
                )
                if len(shibor) > 0:
                    shibor['date'] = pd.to_datetime(shibor['date'])
                    shibor = shibor.rename(columns={
                        'on': 'shibor_on',
                        '1w': 'shibor_1w',
                        '1m': 'shibor_1m'
                    })
                    
                    # 利率是日度数据，需要与月度数据对齐
                    # 取每月最后一天的利率
                    shibor['month'] = shibor['date'].dt.to_period('M')
                    shibor_monthly = shibor.groupby('month').last().reset_index()
                    shibor_monthly['date'] = shibor_monthly['month'].dt.to_timestamp()
                    shibor_monthly = shibor_monthly[['date', 'shibor_on', 'shibor_1w', 'shibor_1m']]
                    
                    # 合并利率数据
                    if len(macro_dfs) > 0:
                        macro_dfs[0] = pd.merge(macro_dfs[0], shibor_monthly, on='date', how='outer')
                    else:
                        macro_dfs.append(shibor_monthly)
                    print(f"    ✓ Shibor数据: {len(shibor_monthly)}条")
            except Exception as e:
                print(f"    加载利率数据失败: {e}")
            
            # 5. GDP数据（季度）
            try:
                print("  加载GDP数据...")
                gdp = self.pro.cn_gdp(
                    start_date=start_date[:6],
                    end_date=end_date[:6]
                )
                if len(gdp) > 0:
                    print(f"    GDP列名: {gdp.columns.tolist()}")
                    print(f"    GDP样例: {gdp['quarter'].head().tolist() if 'quarter' in gdp.columns else 'No quarter column'}")
                    
                    # 季度数据转换：处理多种可能的格式
                    # 可能的格式: '2024Q1', '202401', '2024-Q1' 等
                    if 'quarter' in gdp.columns:
                        # 尝试不同的解析方法
                        try:
                            # 方法1: 如果是 '2024Q1' 格式
                            if gdp['quarter'].iloc[0].endswith(('Q1', 'Q2', 'Q3', 'Q4')):
                                # 转换为季度末日期
                                gdp['date'] = pd.PeriodIndex(gdp['quarter'], freq='Q').to_timestamp(how='end')
                            # 方法2: 如果是 '202401' 格式（YYYYMM）
                            elif len(str(gdp['quarter'].iloc[0])) == 6:
                                gdp['date'] = pd.to_datetime(gdp['quarter'], format='%Y%m')
                                # 转换到季度末
                                gdp['date'] = gdp['date'].dt.to_period('Q').dt.to_timestamp(how='end')
                            else:
                                # 尝试通用解析
                                gdp['date'] = pd.to_datetime(gdp['quarter'])
                                gdp['date'] = gdp['date'].dt.to_period('Q').dt.to_timestamp(how='end')
                        except Exception as parse_error:
                            print(f"    GDP日期解析失败: {parse_error}")
                            # 最后尝试：假设是季度号，手动构造
                            # 提取年份和季度
                            gdp['year'] = gdp['quarter'].str[:4].astype(int)
                            gdp['q'] = gdp['quarter'].str[-1].astype(int)
                            # 构造季度末日期
                            gdp['date'] = pd.to_datetime(gdp['year'].astype(str) + '-' + 
                                                         (gdp['q'] * 3).astype(str) + '-01') + pd.offsets.MonthEnd(0)
                            gdp = gdp.drop(['year', 'q'], axis=1)
                        
                        # 选择需要的列
                        gdp_cols = ['date']
                        if 'gdp' in gdp.columns:
                            gdp_cols.append('gdp')
                        if 'gdp_yoy' in gdp.columns:
                            gdp_cols.append('gdp_yoy')
                        
                        gdp = gdp[gdp_cols]
                        
                        # 合并GDP数据
                        if len(macro_dfs) > 0:
                            macro_dfs[0] = pd.merge(macro_dfs[0], gdp, on='date', how='outer')
                        else:
                            macro_dfs.append(gdp)
                        print(f"    ✓ GDP数据: {len(gdp)}条，列: {gdp_cols}")
                    else:
                        print("    ✗ GDP数据缺少quarter列")
            except Exception as e:
                print(f"    加载GDP数据失败: {e}")
                import traceback
                print(f"    详细错误: {traceback.format_exc()}")

            
            if macro_dfs:
                macro_df = macro_dfs[0]
                macro_df = macro_df.sort_values('date').reset_index(drop=True)
                
                # 前向填充，使得月度/季度数据可以与日度股票数据对齐
                macro_df = macro_df.set_index('date').resample('D').ffill().reset_index()
                
                return macro_df
            else:
                return pd.DataFrame()
        else:
            raise NotImplementedError(f"Macro data not implemented for {self.source}")


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
