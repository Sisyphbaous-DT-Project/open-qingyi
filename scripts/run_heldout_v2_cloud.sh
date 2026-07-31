#!/bin/bash
# 新尺子批量复测：教师 → 蒸馏 v2-750 → CPT+KL-1250（云端 4080S）
cd /root/autodl-tmp/qingyi-kda
PY=/root/miniconda3/bin/python
LOG=logs/heldout_v2_cloud_evals.log
: > "$LOG"
for spec in \
  "teacher:models/Qwen3-0.6B-Base" \
  "distill-v2-750:models/distill-v2-checkpoints/step-750" \
  "cptkl-1250:models/cptkl-checkpoints/step-1250"
do
  tag="${spec%%:*}"
  ck="${spec#*:}"
  echo "===== $tag ($ck) =====" >> "$LOG"
  $PY -u scripts/eval_heldout.py "$ck" --tag "$tag" >> "$LOG" 2>&1
  echo "===== $tag done (exit $?) =====" >> "$LOG"
done
echo "ALL_EVALS_DONE" >> "$LOG"
