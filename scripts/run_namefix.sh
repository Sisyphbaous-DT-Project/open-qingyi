#!/bin/bash
cd /root/projects/qingyi-kda
setsid nohup /root/miniconda3/bin/python scripts/sft.py \
  --dataset data/sft/namefix_dataset.pt \
  --start-from models/dpo-checkpoints/best \
  --save-dir models/namefix-checkpoints \
  --epochs 2 --lr 1e-5 --warmup 10 \
  --eval-every 100 --log-every 10 --n-heldout 60 \
  > logs/namefix.log 2>&1 &
echo "launched pid $!"
