#!/bin/bash
# Milestone test: C-Eval for a step-N checkpoint (+ best-ce) and gen panel.
# Usage: test_milestone.sh <STEP>   e.g. test_milestone.sh 2500
# Runs on the cloud box. C-Eval for step-N and best-ce run in parallel with
# the generation panel.
set -e
STEP=$1
cd /root/autodl-tmp/qingyi-kda

setup_dir () {
  local TAG=$1 SRC=$2
  rm -rf "models/eval-loadable-$TAG"
  mkdir -p "models/eval-loadable-$TAG"
  cp models/hf-skeleton/* "models/eval-loadable-$TAG"/
  sed -i 's/"head_dim": 64/"head_dim": 128/' "models/eval-loadable-$TAG/config.json"
  cp "$SRC/model.safetensors" "$SRC/layout.json" "models/eval-loadable-$TAG"/
}

setup_dir "kd$STEP" "models/kd-checkpoints/step-$STEP"
setup_dir kdbest models/kd-checkpoints/best-ce

export HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1

nohup /root/miniconda3/bin/python -m lm_eval --model hf \
  --model_args pretrained=/root/autodl-tmp/qingyi-kda/models/eval-loadable-kd$STEP,dtype=bfloat16,trust_remote_code=True \
  --tasks ceval-valid --device cuda --batch_size 8 \
  --output_path /root/autodl-tmp/qingyi-kda/eval_results/kd$STEP-ceval \
  > /root/autodl-tmp/kd$STEP-ceval.log 2>&1 &
echo CEVAL_${STEP}_LAUNCHED

nohup /root/miniconda3/bin/python -m lm_eval --model hf \
  --model_args pretrained=/root/autodl-tmp/qingyi-kda/models/eval-loadable-kdbest,dtype=bfloat16,trust_remote_code=True \
  --tasks ceval-valid --device cuda --batch_size 8 \
  --output_path /root/autodl-tmp/qingyi-kda/eval_results/kdbest-ceval \
  > /root/autodl-tmp/kdbest-ceval.log 2>&1 &
echo CEVAL_BEST_LAUNCHED

nohup /root/miniconda3/bin/python scripts/generate_sample.py \
  models/kd-checkpoints/step-$STEP --tokens 80 \
  > /root/autodl-tmp/kd$STEP-gen.txt 2>&1 &
echo GEN_${STEP}_LAUNCHED
