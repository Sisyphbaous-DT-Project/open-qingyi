#!/bin/bash
# Launch end-to-end KL distillation on AutoDL.
# Student start: raw cpt-checkpoints/best (load_hybrid rebuilds from base+layout).
cd /root/projects/qingyi-kda
setsid nohup /root/miniconda3/bin/python -u scripts/kl_distill.py \
  --steps 2000 \
  --start-from /root/projects/qingyi-kda/models/cpt-checkpoints/best \
  --save-dir models/kl-checkpoints \
  --save-every 250 --eval-every 250 --log-every 25 \
  > logs/kl_distill.log 2>&1 &
echo "kl launched pid $!"
