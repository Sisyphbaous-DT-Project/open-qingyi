#!/bin/bash
# Joint CPT + teacher-KL training launcher (cloud, AutoDL 4080S).
# START_FROM: hybrid checkpoint dir (distill-v2 endpoint), override via env.
set -e
cd /root/autodl-tmp/qingyi-kda
export QINGYI_ROOT=/root/autodl-tmp/qingyi-kda
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false

START_FROM=${START_FROM:-models/distill-v2-checkpoints/step-750}

exec /root/miniconda3/bin/python -u scripts/cpt_kl.py \
  --steps 3000 \
  --micro-batch 4 --seq-len 2048 --grad-accum 6 \
  --lr 1e-4 --min-lr 1e-5 --warmup 100 \
  --kl-weight 1.0 --temperature 1.0 \
  --start-from "$START_FROM" \
  --save-every 250 --eval-every 250 --gen-every 500 --log-every 25 \
  --save-dir models/cptkl-checkpoints \
  >> logs/cptkl.log 2>&1
