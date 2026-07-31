#!/usr/bin/env bash
# Launch the production CPT run. Auto-resumes from the newest step checkpoint
# in models/cpt-checkpoints/ if one exists; otherwise cold-starts from the
# distillation best checkpoint (fresh optimizer, step 0).
set -uo pipefail
source /root/projects/qingyi-kda/scripts/env.sh
cd /root/projects/qingyi-kda

CKPT_ROOT=models/cpt-checkpoints
LOG=logs/cpt.log
mkdir -p logs "$CKPT_ROOT"

RESUME=""
latest=$(ls -d "$CKPT_ROOT"/step-* 2>/dev/null | grep -v '\.tmp$' | sort -t- -k2 -n | tail -1 || true)
if [ -n "$latest" ] && [ -f "$latest/meta.json" ] && [ -f "$latest/optimizer.pt" ]; then
  RESUME="--resume $latest"
  echo "[start_cpt] resuming from $latest"
else
  echo "[start_cpt] no CPT checkpoint found, cold start from distillation best"
fi

# shellcheck disable=SC2086
/root/projects/qingyi-kda/.venv/bin/python -u scripts/cpt.py \
  --steps 10000 \
  --micro-batch 4 \
  --seq-len 2048 \
  --grad-accum 6 \
  --lr 3e-4 \
  --min-lr 3e-5 \
  --warmup 100 \
  --weight-decay 0.1 \
  --save-every 250 \
  --eval-every 250 \
  --gen-every 500 \
  --log-every 25 \
  --save-dir "$CKPT_ROOT" \
  $RESUME \
  2>&1 | while IFS= read -r line; do
    printf '%s %s\n' "$(date '+%F %T')" "$line"
  done | tee -a "$LOG"
