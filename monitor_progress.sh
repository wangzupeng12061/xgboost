#!/bin/bash
# 监控程序进度

LOG_FILE="run_2018_2024_100stocks.log"

echo "=========================================="
echo "XGBoost多因子选股系统 - 进度监控"
echo "=========================================="
echo ""
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 检查进程是否在运行
if ps aux | grep "python main.py" | grep -v grep > /dev/null; then
    echo "✅ 程序正在运行"
else
    echo "❌ 程序未运行"
fi

echo ""
echo "--- 最新进度 ---"
tail -20 "$LOG_FILE" | grep -E "(INFO|WARNING|ERROR)" | tail -10

echo ""
echo "--- 关键里程碑 ---"
if grep -q "数据加载完成" "$LOG_FILE" 2>/dev/null; then
    echo "✅ 数据加载完成"
else
    echo "⏳ 数据加载中..."
    # 显示加载进度
    grep "已加载" "$LOG_FILE" 2>/dev/null | tail -3
fi

if grep -q "因子计算完成" "$LOG_FILE" 2>/dev/null; then
    echo "✅ 因子计算完成"
fi

if grep -q "因子筛选完成" "$LOG_FILE" 2>/dev/null; then
    echo "✅ 因子筛选完成"
fi

if grep -q "模型训练完成" "$LOG_FILE" 2>/dev/null; then
    echo "✅ 模型训练完成"
fi

if grep -q "回测完成" "$LOG_FILE" 2>/dev/null; then
    echo "✅ 回测完成"
fi

if grep -q "绩效评估完成" "$LOG_FILE" 2>/dev/null; then
    echo "✅ 绩效评估完成"
fi

if grep -q "程序执行成功" "$LOG_FILE" 2>/dev/null; then
    echo ""
    echo "🎉 全部完成！"
    echo ""
    echo "--- 回测摘要 ---"
    grep -A 10 "回测摘要" "$LOG_FILE" | tail -10
fi

echo ""
echo "=========================================="
