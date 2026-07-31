#!/usr/bin/env bash
# Launch the production distillation run. Auto-resumes from the newest step
# checkpoint in models/distill-checkpoints/ if one exists.
set -uo pipefail
source /root/projects/qingyi-kda/scripts/env.sh
cd /root/projects/qingyi-kda

CKPT_ROOT=models/distill-checkpoints
LOG=logs/distill.log
mkdir -p logs "$CKPT_ROOT"

# Find the newest complete step checkpoint (has meta.json + optimizer.pt).
RESUME=""
latest=$(ls -d "$CKPT_ROOT"/step-* 2>/dev/null | grep -v '\.tmp$' | sort -t- -k2 -n | tail -1 || true)
if [ -n "$latest" ] && [ -f "$latest/meta.json" ] && [ -f "$latest/optimizer.pt" ]; then
  RESUME="--resume $latest"
  echo "[start_distill] resuming from $latest"
else
  echo "[start_distill] no checkpoint found, starting from scratch"
fi

# shellcheck disable=SC2086
# -u: unbuffered stdout so the timestamped log stays current through the pipe.
/root/projects/qingyi-kda/.venv/bin/python -u scripts/distill.py \
  --steps 3500 \
  --micro-batch 4 \
  --seq-len 2048 \
  --grad-accum 6 \
  --lr 1e-3 \
  --min-lr 1e-4 \
  --warmup 100 \
  --save-every 100 \
  --eval-every 250 \
  --eval-batches 2 \
  --log-every 25 \
  --save-dir "$CKPT_ROOT" \
  $RESUME \
  2>&1 | while IFS= read -r line; do
    printf '%s %s\n' "$(date '+%F %T')" "$line"
  done | tee -a "$LOG"
