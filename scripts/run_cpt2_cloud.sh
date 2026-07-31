#!/bin/bash
# CPT 加量续训：从 cpt-checkpoints/best 冷启动优化器，追加 3000 步
set -e
cd /root/autodl-tmp/qingyi-kda
mv /root/autodl-tmp/qingyi-kda/logs/cpt2.log /root/autodl-tmp/qingyi-kda/logs/cpt2_$(date +%m%d_%H%M).log 2>/dev/null || true
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /root/miniconda3/bin/python -u scripts/cpt.py \
  --start-from /root/autodl-tmp/qingyi-kda/models/cpt-checkpoints/best \
  --steps 3000 \
  --micro-batch 4 --grad-accum 6 --no-grad-ckpt \
  --lr 1e-4 --min-lr 1e-5 --warmup 50 \
  --save-dir /root/autodl-tmp/qingyi-kda/models/cpt2-checkpoints \
  --save-every 250 --eval-every 250 --gen-every 500 --log-every 25 \
  >> /root/autodl-tmp/qingyi-kda/logs/cpt2.log 2>&1
touch /root/autodl-tmp/qingyi-kda/logs/cpt2.done
