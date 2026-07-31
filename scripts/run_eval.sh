#!/bin/bash
# Run lm-evaluation-harness on a model.
# Usage: run_eval.sh <model_path> <tasks> <tag> [num_fewshot]
set -e
cd /root/projects/qingyi-kda
source scripts/proxy_env.sh >/dev/null
MODEL=$1
TASKS=$2
TAG=$3
BATCH=${4:-auto}
OUT=eval_results/$TAG
mkdir -p "$OUT"
.venv/bin/python -m lm_eval \
  --model hf \
  --model_args "pretrained=$MODEL,dtype=bfloat16,trust_remote_code=True" \
  --tasks "$TASKS" \
  --device cuda \
  --batch_size "$BATCH" \
  --output_path "$OUT" \
  --log_samples 2>&1 | tee "$OUT/run.log"
