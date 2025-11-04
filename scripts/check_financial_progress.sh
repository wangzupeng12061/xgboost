#!/bin/bash
# 检查财务数据下载进度

echo "=========================================="
echo "财务数据下载进度监控"
echo "=========================================="

# 检查进程是否在运行
if ps aux | grep -v grep | grep "download_other_data.py" > /dev/null; then
    echo "✓ 下载进程正在运行"
else
    echo "✗ 下载进程未运行"
fi

echo ""
echo "已下载文件数量:"
DOWNLOADED=$(ls -1 data/financial/ 2>/dev/null | wc -l | tr -d ' ')
echo "  $DOWNLOADED / 5445 ($(echo "scale=1; $DOWNLOADED * 100 / 5445" | bc)%)"

echo ""
echo "数据大小:"
du -sh data/financial/ 2>/dev/null

echo ""
echo "最近日志 (最后10条):"
tail -10 logs/download_other_data_*.log 2>/dev/null | grep -E "(进度|下载成功|完成)" | tail -5

echo ""
echo "=========================================="
echo "提示: 使用 'tail -f logs/financial_download.log' 查看实时日志"
echo "=========================================="
