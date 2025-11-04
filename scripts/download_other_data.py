"""
下载其他必要数据：指数、宏观、财务数据
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
import pandas as pd
import tushare as ts
from src.data.data_cache import DataCache
from src.utils.logger import setup_logger
import yaml

logger = setup_logger("download_other_data")


class OtherDataDownloader:
    """其他数据下载器"""

    def __init__(self, token: str, cache_dir: str = "./data"):
        ts.set_token(token)
        self.pro = ts.pro_api()
        self.cache = DataCache(cache_dir=cache_dir, expire_days=0)
        self.api_wait = 0.5  # API调用间隔

    def download_index_data(self, start_date: str, end_date: str):
        """下载指数数据"""
        logger.info("=" * 70)
        logger.info("开始下载指数数据...")
        logger.info("=" * 70)

        # 主要指数列表
        indices = {
            "000300.SH": "沪深300",
            "000905.SH": "中证500",
            "000001.SH": "上证指数",
            "399001.SZ": "深证成指",
            "399006.SZ": "创业板指",
        }

        success = 0
        fail = 0

        for ts_code, name in indices.items():
            try:
                # 检查缓存
                cached = self.cache.get_index_daily(ts_code)
                if cached is not None:
                    logger.info(f"✓ {name} ({ts_code}) - 已缓存，跳过")
                    continue

                logger.info(f"下载 {name} ({ts_code})...")
                time.sleep(self.api_wait)

                # 下载数据
                df = self.pro.index_daily(
                    ts_code=ts_code,
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                )

                if df is not None and len(df) > 0:
                    self.cache.save_index_daily(ts_code, df)
                    logger.info(f"✓ {name} ({ts_code}) 下载成功，数据量: {len(df)}")
                    success += 1
                else:
                    logger.warning(f"✗ {name} ({ts_code}) 无数据")
                    fail += 1

            except Exception as e:
                logger.error(f"✗ {name} ({ts_code}) 下载失败: {e}")
                fail += 1
                time.sleep(5)

        logger.info(f"\n指数数据下载完成: 成功 {success}, 失败 {fail}")

    def download_macro_data(self):
        """下载宏观数据"""
        logger.info("\n" + "=" * 70)
        logger.info("开始下载宏观数据...")
        logger.info("=" * 70)

        macro_indicators = {
            "m1": {"name": "M1货币供应量", "func": "cn_m"},
            "m2": {"name": "M2货币供应量", "func": "cn_m"},
            "cpi": {"name": "CPI", "func": "cn_cpi"},
            "ppi": {"name": "PPI", "func": "cn_ppi"},
            "gdp": {"name": "GDP", "func": "cn_gdp"},
            "pmi": {"name": "PMI", "func": "cn_pmi"},
        }

        success = 0
        fail = 0

        for indicator, info in macro_indicators.items():
            try:
                # 检查缓存
                cached = self.cache.get_macro_data(indicator)
                if cached is not None:
                    logger.info(f"✓ {info['name']} - 已缓存，跳过")
                    continue

                logger.info(f"下载 {info['name']}...")
                time.sleep(self.api_wait)

                # 根据不同指标调用不同接口
                if indicator == "m1":
                    df = self.pro.cn_m(m_type="1")
                elif indicator == "m2":
                    df = self.pro.cn_m(m_type="2")
                elif indicator == "cpi":
                    df = self.pro.cn_cpi()
                elif indicator == "ppi":
                    df = self.pro.cn_ppi()
                elif indicator == "gdp":
                    df = self.pro.cn_gdp()
                elif indicator == "pmi":
                    df = self.pro.cn_pmi()
                else:
                    continue

                if df is not None and len(df) > 0:
                    self.cache.save_macro_data(indicator, df)
                    logger.info(f"✓ {info['name']} 下载成功，数据量: {len(df)}")
                    success += 1
                else:
                    logger.warning(f"✗ {info['name']} 无数据")
                    fail += 1

            except Exception as e:
                logger.error(f"✗ {info['name']} 下载失败: {e}")
                fail += 1
                time.sleep(5)

        logger.info(f"\n宏观数据下载完成: 成功 {success}, 失败 {fail}")

    def download_stock_basic(self):
        """下载股票基本信息"""
        logger.info("\n" + "=" * 70)
        logger.info("开始下载股票基本信息...")
        logger.info("=" * 70)

        try:
            # 检查缓存
            cached = self.cache.get_stock_basic()
            if cached is not None:
                logger.info(f"✓ 股票基本信息 - 已缓存，跳过")
                return

            logger.info("下载股票基本信息...")
            time.sleep(self.api_wait)

            df = self.pro.stock_basic(
                exchange="",
                list_status="L",
                fields="ts_code,symbol,name,area,industry,market,list_date",
            )

            if df is not None and len(df) > 0:
                self.cache.save_stock_basic(df)
                logger.info(f"✓ 股票基本信息下载成功，数据量: {len(df)}")
            else:
                logger.warning("✗ 股票基本信息无数据")

        except Exception as e:
            logger.error(f"✗ 股票基本信息下载失败: {e}")

    def download_financial_data(self, stock_list: list, start_date: str, end_date: str):
        """下载财务数据（利润表、资产负债表、现金流量表）"""
        logger.info("\n" + "=" * 70)
        logger.info("开始下载财务数据...")
        logger.info(f"股票数量: {len(stock_list)}")
        logger.info("=" * 70)

        success = 0
        fail = 0
        skipped = 0

        # 转换日期格式
        start = start_date.replace("-", "")
        end = end_date.replace("-", "")

        for i, ts_code in enumerate(stock_list, 1):
            try:
                # 检查缓存
                cached = self.cache.get_financial_data(ts_code)
                if cached is not None and len(cached) > 0:
                    logger.info(f"[{i}/{len(stock_list)}] {ts_code} - 已缓存，跳过")
                    skipped += 1
                    continue

                logger.info(f"[{i}/{len(stock_list)}] 下载 {ts_code} 财务数据...")
                time.sleep(self.api_wait)

                # 下载财务数据（包含三大报表的关键指标）
                df = self.pro.fina_indicator(
                    ts_code=ts_code,
                    start_date=start,
                    end_date=end,
                    fields='ts_code,ann_date,end_date,eps,dt_eps,total_revenue_ps,revenue_ps,capital_rese_ps,'
                           'surplus_rese_ps,undist_profit_ps,extra_item,profit_dedt,gross_margin,current_ratio,'
                           'quick_ratio,cash_ratio,invturn_days,arturn_days,inv_turn,ar_turn,ca_turn,fa_turn,'
                           'assets_turn,op_income,valuechange_income,ebit,ebitda,fcff,fcfe,current_exint,'
                           'noncurrent_exint,interestdebt,netdebt,tangible_asset,working_capital,networking_capital,'
                           'invest_capital,retained_earnings,diluted2_eps,bps,ocfps,retainedps,cfps,ebit_ps,'
                           'fcff_ps,fcfe_ps,netprofit_margin,grossprofit_margin,cogs_of_sales,expense_of_sales,'
                           'profit_to_gr,saleexp_to_gr,adminexp_of_gr,finaexp_of_gr,impai_ttm,gc_of_gr,'
                           'op_of_gr,ebit_of_gr,roe,roe_waa,roe_dt,roa,npta,roic,roe_yearly,roa_yearly,'
                           'roe_avg,opincome_of_ebt,investincome_of_ebt,n_op_profit_of_ebt,tax_to_ebt,'
                           'dtprofit_to_profit,salescash_to_or,ocf_to_or,ocf_to_opincome,capitalized_to_da,'
                           'debt_to_assets,assets_to_eqt,dp_assets_to_eqt,ca_to_assets,nca_to_assets,'
                           'tbassets_to_totalassets,int_to_talcap,eqt_to_talcapital,currentdebt_to_debt,'
                           'longdeb_to_debt,ocf_to_shortdebt,debt_to_eqt,eqt_to_debt,eqt_to_interestdebt,'
                           'tangibleasset_to_debt,tangasset_to_intdebt,tangibleasset_to_netdebt,ocf_to_debt,'
                           'ocf_to_interestdebt,ocf_to_netdebt,ebit_to_interest,longdebt_to_workingcapital,'
                           'ebitda_to_debt,turn_days,roa_yearly,roa_dp,fixed_assets,profit_prefin_exp,'
                           'non_op_profit,op_to_ebt,nop_to_ebt,ocf_to_profit,cash_to_liqdebt,cash_to_liqdebt_withinterest,'
                           'op_to_liqdebt,op_to_debt,roic_yearly,total_fa_trun,profit_to_op,q_opincome,'
                           'q_investincome,q_dtprofit,q_eps,q_netprofit_margin,q_gsprofit_margin,q_exp_to_sales,'
                           'q_profit_to_gr,q_saleexp_to_gr,q_adminexp_to_gr,q_finaexp_to_gr,q_impair_to_gr_ttm,'
                           'q_gc_to_gr,q_op_to_gr,q_roe,q_dt_roe,q_npta,q_opincome_to_ebt,q_investincome_to_ebt,'
                           'q_dtprofit_to_profit,q_salescash_to_or,q_ocf_to_sales,q_ocf_to_or,basic_eps_yoy,'
                           'dt_eps_yoy,cfps_yoy,op_yoy,ebt_yoy,netprofit_yoy,dt_netprofit_yoy,ocf_yoy,roe_yoy,'
                           'bps_yoy,assets_yoy,eqt_yoy,tr_yoy,or_yoy,q_gr_yoy,q_gr_qoq,q_sales_yoy,q_sales_qoq,'
                           'q_op_yoy,q_op_qoq,q_profit_yoy,q_profit_qoq,q_netprofit_yoy,q_netprofit_qoq,'
                           'equity_yoy,rd_exp,update_flag'
                )

                if df is not None and len(df) > 0:
                    self.cache.save_financial_data(ts_code, df)
                    logger.info(f"✓ {ts_code} 下载成功，数据量: {len(df)}")
                    success += 1
                else:
                    logger.warning(f"✗ {ts_code} 无数据")
                    fail += 1

                # 每100个股票打印一次进度
                if i % 100 == 0:
                    logger.info(f"进度: {i}/{len(stock_list)} ({i*100//len(stock_list)}%)")

            except Exception as e:
                logger.error(f"✗ {ts_code} 下载失败: {e}")
                fail += 1
                time.sleep(5)

        logger.info(f"\n财务数据下载完成: 成功 {success}, 跳过 {skipped}, 失败 {fail}")

    def download_all(self, start_date: str, end_date: str, download_financial: bool = True):
        """下载所有数据"""
        logger.info("\n" + "=" * 70)
        logger.info("开始下载其他必要数据")
        logger.info(f"日期范围: {start_date} 至 {end_date}")
        logger.info("=" * 70)

        # 1. 股票基本信息
        self.download_stock_basic()

        # 2. 指数数据
        self.download_index_data(start_date, end_date)

        # 3. 宏观数据
        self.download_macro_data()

        # 4. 财务数据（可选）
        if download_financial:
            # 获取股票列表
            stock_basic = self.cache.get_stock_basic()
            if stock_basic is not None:
                stock_list = stock_basic['ts_code'].tolist()
                logger.info(f"\n找到 {len(stock_list)} 只股票，准备下载财务数据...")
                self.download_financial_data(stock_list, start_date, end_date)
            else:
                logger.warning("无法获取股票列表，跳过财务数据下载")

        # 最终统计
        logger.info("\n" + "=" * 70)
        logger.info("所有数据下载完成！")
        logger.info("=" * 70)

        stats = self.cache.get_cache_stats()
        logger.info(f"\n缓存统计:")
        logger.info(f"  股票日线: {stats['stock_daily_count']} 个文件")
        logger.info(f"  股票基本信息: {'已缓存' if stats['stock_basic_exists'] else '未缓存'}")
        logger.info(f"  指数数据: {stats['index_daily_count']} 个文件")
        logger.info(f"  宏观数据: {stats['macro_count']} 个文件")
        logger.info(f"  财务数据: {stats['financial_count']} 个文件")
        logger.info(f"  总大小: {stats['total_size_mb']} MB")
        logger.info("=" * 70)


def main():
    # 读取配置
    try:
        with open("config/config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            token = config["data"]["token"]
    except Exception as e:
        logger.error(f"无法读取配置: {e}")
        sys.exit(1)

    # 创建下载器
    downloader = OtherDataDownloader(token=token, cache_dir="./data")

    # 下载数据
    start_date = "2020-01-01"
    end_date = "2025-11-04"
    downloader.download_all(start_date, end_date)

    logger.info("\n✓ 全部完成!")


if __name__ == "__main__":
    main()
