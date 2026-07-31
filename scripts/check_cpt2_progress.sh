#!/bin/bash
# 诊断 cpt2 训练进程：CPU 时间增量 + GPU + triton 编译缓存新鲜度 + 检查点
PID=$1
a=$(awk '{print $14}' /proc/$PID/stat)
sleep 45
b=$(awk '{print $14}' /proc/$PID/stat)
echo "CPU-JIFFIES-DELTA:$((b-a)) (100 jiffies = 1s CPU)"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "--- triton cache files touched in last 3 min:"
find /root/.triton/cache -newermt '-3 minutes' 2>/dev/null | head -3
echo "--- checkpoints:"
ls /root/autodl-tmp/qingyi-kda/models/cpt2-checkpoints/ 2>/dev/null
echo "--- log size:"
wc -c /root/autodl-tmp/qingyi-kda/logs/cpt2.log
