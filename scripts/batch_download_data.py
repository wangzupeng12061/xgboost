"""
批量下载多市场股票数据脚本
支持A股、港股、美股等多个市场，分批次下载数据

使用方法:
    python scripts/batch_download_data.py --market a --batch-size 50 --start-date 2020-01-01 --end-date 2025-11-04
    python scripts/batch_download_data.py --market all --batch-size 50 --total 1000
"""

import sys
import os
import time
import argparse
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import tushare as ts
from src.data.data_cache import DataCache
from src.utils.logger import setup_logger

# 设置日志
logger = setup_logger("batch_download")


class BatchDataDownloader:
    """批量数据下载器"""

    def __init__(self, token: str, cache_dir: str = "./data"):
        """
        初始化下载器

        Args:
            token: Tushare token
            cache_dir: 缓存目录
        """
        ts.set_token(token)
        self.pro = ts.pro_api()
        self.cache = DataCache(cache_dir=cache_dir, expire_days=0)  # 永不过期
        
        # API限流控制
        self.api_call_interval = 0.5  # 每次API调用间隔（秒）
        self.batch_wait_time = 60  # 批次间等待时间（秒）
        self.api_calls_count = 0  # API调用计数
        self.api_calls_limit = 180  # 每分钟API调用限制

    def _api_call_with_retry(self, func, max_retries=3, **kwargs):
        """
        带重试的API调用

        Args:
            func: API函数
            max_retries: 最大重试次数
            **kwargs: API参数

        Returns:
            API返回的DataFrame
        """
        for retry in range(max_retries):
            try:
                # 检查API调用频率
                if self.api_calls_count >= self.api_calls_limit:
                    logger.warning(f"达到API调用限制({self.api_calls_limit}次/分钟)，等待{self.batch_wait_time}秒...")
                    time.sleep(self.batch_wait_time)
                    self.api_calls_count = 0

                # API调用
                time.sleep(self.api_call_interval)
                result = func(**kwargs)
                self.api_calls_count += 1

                if result is not None and len(result) > 0:
                    return result
                else:
                    logger.warning(f"API返回空数据，重试 {retry + 1}/{max_retries}")

            except Exception as e:
                error_msg = str(e)
                
                # 检测限流错误
                if "抱歉，您每分钟最多访问" in error_msg or "接口限流" in error_msg:
                    logger.warning(f"触发API限流，等待{self.batch_wait_time}秒...")
                    time.sleep(self.batch_wait_time)
                    self.api_calls_count = 0
                else:
                    logger.error(f"API调用失败: {error_msg}, 重试 {retry + 1}/{max_retries}")
                    
                if retry < max_retries - 1:
                    time.sleep(5)  # 重试前等待5秒

        logger.error(f"API调用失败，已达最大重试次数")
        return None

    def get_stock_list(self, market: str = "a", total: int = 1000) -> pd.DataFrame:
        """
        获取股票列表

        Args:
            market: 市场类型 ('a'=A股, 'hk'=港股, 'us'=美股, 'all'=全部)
            total: 获取数量

        Returns:
            股票列表DataFrame
        """
        logger.info(f"获取{market}市场股票列表，目标数量: {total}")

        all_stocks = []

        # A股
        if market in ["a", "all"]:
            logger.info("获取A股列表...")
            a_stocks = self._api_call_with_retry(
                self.pro.stock_basic,
                exchange="",
                list_status="L",
                fields="ts_code,symbol,name,area,industry,market,list_date"
            )
            if a_stocks is not None:
                a_stocks["market_type"] = "A股"
                all_stocks.append(a_stocks)
                logger.info(f"✓ A股数量: {len(a_stocks)}")

        # 港股
        if market in ["hk", "all"]:
            logger.info("获取港股列表...")
            hk_stocks = self._api_call_with_retry(
                self.pro.hk_basic,
                list_status="L",
                fields="ts_code,name,fullname,enname,market,list_date"
            )
            if hk_stocks is not None:
                hk_stocks["market_type"] = "港股"
                all_stocks.append(hk_stocks)
                logger.info(f"✓ 港股数量: {len(hk_stocks)}")

        # 美股
        if market in ["us", "all"]:
            logger.info("获取美股列表...")
            # 美股数据获取（Tushare需要特殊权限）
            try:
                us_stocks = self._api_call_with_retry(
                    self.pro.us_basic,
                    fields="ts_code,name,market,list_date"
                )
                if us_stocks is not None:
                    us_stocks["market_type"] = "美股"
                    all_stocks.append(us_stocks)
                    logger.info(f"✓ 美股数量: {len(us_stocks)}")
            except Exception as e:
                logger.warning(f"美股数据获取失败（可能需要更高权限）: {e}")

        # 合并所有股票
        if not all_stocks:
            logger.error("未能获取任何股票列表")
            return pd.DataFrame()

        df_all = pd.concat(all_stocks, ignore_index=True)
        
        # 按市值排序（如果有市值数据）
        # 这里简单按上市时间排序，优先获取老牌股票
        df_all = df_all.sort_values("list_date")
        
        # 限制数量
        df_all = df_all.head(total)
        
        logger.info(f"✓ 总计获取 {len(df_all)} 只股票")
        return df_all

    def download_stock_daily_data(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        market_type: str = "A股"
    ) -> bool:
        """
        下载单只股票的日线数据

        Args:
            ts_code: 股票代码
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD
            market_type: 市场类型

        Returns:
            bool: 是否成功
        """
        # 检查缓存
        cached_data = self.cache.get_stock_daily(ts_code)
        if cached_data is not None:
            logger.debug(f"✓ {ts_code} 已有缓存，跳过")
            return True

        # 转换日期格式
        start = start_date.replace("-", "")
        end = end_date.replace("-", "")

        logger.info(f"下载 {ts_code} ({market_type}) 数据: {start_date} 至 {end_date}")

        # 根据市场类型选择API
        if market_type == "A股":
            df = self._api_call_with_retry(
                self.pro.daily,
                ts_code=ts_code,
                start_date=start,
                end_date=end
            )
        elif market_type == "港股":
            df = self._api_call_with_retry(
                self.pro.hk_daily,
                ts_code=ts_code,
                start_date=start,
                end_date=end
            )
        elif market_type == "美股":
            df = self._api_call_with_retry(
                self.pro.us_daily,
                ts_code=ts_code,
                start_date=start,
                end_date=end
            )
        else:
            logger.error(f"不支持的市场类型: {market_type}")
            return False

        if df is None or len(df) == 0:
            logger.warning(f"✗ {ts_code} 无数据")
            return False

        # 保存到缓存
        self.cache.save_stock_daily(ts_code, df)
        logger.info(f"✓ {ts_code} 下载成功，数据量: {len(df)}")
        return True

    def batch_download(
        self,
        stock_list: pd.DataFrame,
        start_date: str,
        end_date: str,
        batch_size: int = 50
    ):
        """
        批量下载股票数据

        Args:
            stock_list: 股票列表
            start_date: 开始日期
            end_date: 结束日期
            batch_size: 批次大小
        """
        total_stocks = len(stock_list)
        total_batches = (total_stocks + batch_size - 1) // batch_size

        logger.info("=" * 70)
        logger.info(f"批量下载开始")
        logger.info(f"总股票数: {total_stocks}")
        logger.info(f"批次大小: {batch_size}")
        logger.info(f"总批次数: {total_batches}")
        logger.info(f"日期范围: {start_date} 至 {end_date}")
        logger.info("=" * 70)

        success_count = 0
        skip_count = 0
        fail_count = 0

        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, total_stocks)
            batch_stocks = stock_list.iloc[batch_start:batch_end]

            logger.info("")
            logger.info("=" * 70)
            logger.info(f"批次 {batch_idx + 1}/{total_batches}")
            logger.info(f"处理股票: {batch_start + 1}-{batch_end}/{total_stocks}")
            logger.info("=" * 70)

            batch_start_time = time.time()

            for idx, row in batch_stocks.iterrows():
                ts_code = row["ts_code"]
                market_type = row.get("market_type", "A股")

                # 检查缓存
                cached_data = self.cache.get_stock_daily(ts_code)
                if cached_data is not None:
                    skip_count += 1
                    logger.debug(f"[{idx - batch_start + 1}/{len(batch_stocks)}] {ts_code} - 已缓存，跳过")
                    continue

                # 下载数据
                success = self.download_stock_daily_data(
                    ts_code, start_date, end_date, market_type
                )

                if success:
                    success_count += 1
                else:
                    fail_count += 1

            batch_elapsed = time.time() - batch_start_time
            logger.info(f"批次 {batch_idx + 1} 完成，耗时: {batch_elapsed:.1f}秒")

            # 批次间等待（除了最后一批）
            if batch_idx < total_batches - 1:
                logger.info(f"等待 {self.batch_wait_time} 秒后继续下一批次...")
                time.sleep(self.batch_wait_time)
                self.api_calls_count = 0  # 重置计数

        # 统计信息
        logger.info("")
        logger.info("=" * 70)
        logger.info("下载完成!")
        logger.info(f"总计: {total_stocks} 只股票")
        logger.info(f"成功: {success_count} 只")
        logger.info(f"跳过: {skip_count} 只 (已有缓存)")
        logger.info(f"失败: {fail_count} 只")
        logger.info("=" * 70)

        # 缓存统计
        stats = self.cache.get_cache_stats()
        logger.info("")
        logger.info("缓存统计:")
        logger.info(f"  股票数据文件: {stats['stock_daily_count']}")
        logger.info(f"  总大小: {stats['total_size_mb']} MB")
        logger.info("=" * 70)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="批量下载多市场股票数据")
    
    parser.add_argument(
        "--market",
        type=str,
        default="a",
        choices=["a", "hk", "us", "all"],
        help="市场类型: a=A股, hk=港股, us=美股, all=全部 (默认: a)"
    )
    
    parser.add_argument(
        "--total",
        type=int,
        default=1000,
        help="下载股票数量 (默认: 1000)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="每批次下载数量 (默认: 50)"
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        default="2020-01-01",
        help="开始日期 YYYY-MM-DD (默认: 2020-01-01)"
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-11-04",
        help="结束日期 YYYY-MM-DD (默认: 2025-11-04)"
    )
    
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Tushare token (如不提供，从config.yaml读取)"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./data",
        help="缓存目录 (默认: ./data)"
    )

    args = parser.parse_args()

    # 获取token
    token = args.token
    if token is None:
        try:
            import yaml
            with open("config/config.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                token = config["data"]["token"]
        except Exception as e:
            logger.error(f"无法从config.yaml读取token: {e}")
            logger.error("请使用 --token 参数提供Tushare token")
            sys.exit(1)

    # 创建下载器
    downloader = BatchDataDownloader(token=token, cache_dir=args.cache_dir)

    # 获取股票列表
    logger.info("步骤 1/2: 获取股票列表...")
    stock_list = downloader.get_stock_list(market=args.market, total=args.total)

    if len(stock_list) == 0:
        logger.error("未能获取股票列表，退出")
        sys.exit(1)

    # 保存股票列表
    list_file = f"data/stock_list_{args.market}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    stock_list.to_csv(list_file, index=False, encoding="utf-8-sig")
    logger.info(f"股票列表已保存到: {list_file}")

    # 批量下载
    logger.info("")
    logger.info("步骤 2/2: 批量下载数据...")
    downloader.batch_download(
        stock_list=stock_list,
        start_date=args.start_date,
        end_date=args.end_date,
        batch_size=args.batch_size
    )

    logger.info("")
    logger.info("✓ 所有任务完成!")


if __name__ == "__main__":
    main()
