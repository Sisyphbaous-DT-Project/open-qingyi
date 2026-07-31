#!/bin/bash
# Chain MMLU (and optionally CMMLU) over the three models sequentially.
cd /root/projects/qingyi-kda
TASKS=${1:-mmlu}
for ENTRY in \
  "/root/projects/qingyi-kda/models/Qwen3-0.6B-Base|base" \
  "/root/projects/qingyi-kda/models/cpt-best-loadable|cpt" \
  "/root/projects/qingyi-kda/models/qingyi-kda-0.6b-hf|final"; do
  MODEL=${ENTRY%%|*}
  TAG=${ENTRY##*|}
  echo "=== $TAG @ $MODEL ($TASKS) ==="
  bash scripts/run_eval.sh "$MODEL" "$TASKS" "$TAG-$TASKS" 8
done
echo "ALL-DONE"
