#!/bin/bash
# 批量下载股票数据快速启动脚本

echo "======================================================================"
echo "批量下载多市场股票数据"
echo "======================================================================"
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3"
    exit 1
fi

# 创建scripts目录（如果不存在）
mkdir -p scripts

# 默认参数
MARKET="a"
TOTAL=1000
BATCH_SIZE=50
START_DATE="2020-01-01"
END_DATE="2025-11-04"
CACHE_DIR="./data"

# 显示菜单
echo "请选择市场类型:"
echo "  1) A股 (默认)"
echo "  2) 港股"
echo "  3) 美股"
echo "  4) 全部市场"
echo ""
read -p "请输入选项 [1-4] (默认: 1): " market_choice

case $market_choice in
    2) MARKET="hk" ;;
    3) MARKET="us" ;;
    4) MARKET="all" ;;
    *) MARKET="a" ;;
esac

echo ""
read -p "下载股票数量 (默认: 1000): " total_input
if [ ! -z "$total_input" ]; then
    TOTAL=$total_input
fi

echo ""
read -p "每批次数量 (默认: 50): " batch_input
if [ ! -z "$batch_input" ]; then
    BATCH_SIZE=$batch_input
fi

echo ""
read -p "开始日期 YYYY-MM-DD (默认: 2020-01-01): " start_input
if [ ! -z "$start_input" ]; then
    START_DATE=$start_input
fi

echo ""
read -p "结束日期 YYYY-MM-DD (默认: 2025-11-04): " end_input
if [ ! -z "$end_input" ]; then
    END_DATE=$end_input
fi

echo ""
echo "======================================================================"
echo "配置确认:"
echo "  市场类型: $MARKET"
echo "  股票数量: $TOTAL"
echo "  批次大小: $BATCH_SIZE"
echo "  日期范围: $START_DATE 至 $END_DATE"
echo "  缓存目录: $CACHE_DIR"
echo "======================================================================"
echo ""
read -p "确认开始下载? [y/N]: " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "开始下载..."
echo ""

# 执行Python脚本
python3 scripts/batch_download_data.py \
    --market "$MARKET" \
    --total "$TOTAL" \
    --batch-size "$BATCH_SIZE" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --cache-dir "$CACHE_DIR"

echo ""
echo "======================================================================"
echo "下载完成!"
echo "======================================================================"
