#!/bin/bash
# v2 蒸馏启动器（云上）：teacher-init 手术 + 与 v1 完全相同的蒸馏配置
# 干净消融：唯一变量 = 初始化（teacher-init vs v1 随机）
set -e
cd /root/autodl-tmp/qingyi-kda
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/root/miniconda3/bin/python -u scripts/distill.py \
  --steps 2800 \
  --micro-batch 4 \
  --seq-len 2048 \
  --grad-accum 6 \
  --lr 1e-3 \
  --min-lr 1e-4 \
  --warmup 100 \
  --save-every 250 \
  --eval-every 250 \
  --eval-batches 2 \
  --log-every 25 \
  --save-dir /root/autodl-tmp/qingyi-kda/models/distill-v2-checkpoints \
  >> /root/autodl-tmp/qingyi-kda/logs/distill_v2.log 2>&1
touch /root/autodl-tmp/qingyi-kda/logs/distill_v2.done
