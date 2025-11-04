"""
批量下载测试脚本 - 快速验证功能
下载10只A股的2024年数据进行测试
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_cache import DataCache
from src.utils.logger import setup_logger
import yaml

logger = setup_logger("test_download")

def test_cache():
    """测试缓存功能"""
    logger.info("=" * 70)
    logger.info("测试1: 缓存功能")
    logger.info("=" * 70)
    
    cache = DataCache(cache_dir="./data", expire_days=0)
    stats = cache.get_cache_stats()
    
    logger.info(f"✓ 缓存目录: {stats['cache_dir']}")
    logger.info(f"✓ 已缓存股票数: {stats['stock_daily_count']}")
    logger.info(f"✓ 缓存大小: {stats['total_size_mb']} MB")
    
    return True

def test_config():
    """测试配置读取"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("测试2: 配置读取")
    logger.info("=" * 70)
    
    try:
        with open("config/config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        token = config["data"]["token"]
        cache_enabled = config["data"].get("cache", {}).get("enabled", False)
        
        logger.info(f"✓ Token: {token[:10]}..." if token else "✗ Token未配置")
        logger.info(f"✓ 缓存已{'启用' if cache_enabled else '禁用'}")
        
        return token is not None
    except Exception as e:
        logger.error(f"✗ 配置读取失败: {e}")
        return False

def run_quick_test():
    """运行快速测试下载"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("测试3: 快速下载测试")
    logger.info("=" * 70)
    logger.info("将下载10只A股的2024年数据（约1-2分钟）")
    logger.info("")
    
    import subprocess
    
    cmd = [
        "python", "scripts/batch_download_data.py",
        "--market", "a",
        "--total", "10",
        "--batch-size", "10",
        "--start-date", "2024-01-01",
        "--end-date", "2024-10-31"
    ]
    
    logger.info(f"执行命令: {' '.join(cmd)}")
    logger.info("")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        logger.info("")
        logger.info("✓ 快速测试完成!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ 测试失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("批量下载功能测试")
    logger.info("=" * 70)
    
    results = []
    
    # 测试1: 缓存
    results.append(("缓存功能", test_cache()))
    
    # 测试2: 配置
    results.append(("配置读取", test_config()))
    
    # 测试3: 下载（可选）
    logger.info("")
    response = input("是否运行快速下载测试？(y/N): ")
    if response.lower() == 'y':
        results.append(("快速下载", run_quick_test()))
    else:
        logger.info("跳过下载测试")
        results.append(("快速下载", None))
    
    # 汇总结果
    logger.info("")
    logger.info("=" * 70)
    logger.info("测试结果汇总")
    logger.info("=" * 70)
    
    for name, result in results:
        if result is None:
            status = "⊘ 跳过"
        elif result:
            status = "✓ 通过"
        else:
            status = "✗ 失败"
        logger.info(f"{status} - {name}")
    
    logger.info("=" * 70)
    
    # 显示下一步提示
    if all(r is not False for r in [r[1] for r in results]):
        logger.info("")
        logger.info("🎉 所有测试通过！可以开始批量下载了：")
        logger.info("")
        logger.info("快速开始:")
        logger.info("  ./scripts/download_data.sh")
        logger.info("")
        logger.info("或直接运行:")
        logger.info("  python scripts/batch_download_data.py --market a --total 1000")
        logger.info("")

if __name__ == "__main__":
    main()
