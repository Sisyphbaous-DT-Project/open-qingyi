#!/bin/bash
# Quick C-Eval check on a KL checkpoint: build loadable dir from skeleton + weights, eval.
# usage: kl_eval_ckpt.sh <ckpt_dir> <tag>
set -e
cd /root/projects/qingyi-kda
CKPT=$1
TAG=$2
LOADABLE=models/eval-loadable-$TAG
rm -rf "$LOADABLE"
mkdir -p "$LOADABLE"
cp models/hf-skeleton/* "$LOADABLE"/
cp "$CKPT/model.safetensors" "$LOADABLE"/
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p eval_results
/root/miniconda3/bin/python -m lm_eval \
  --model hf \
  --model_args "pretrained=$LOADABLE,dtype=bfloat16,trust_remote_code=True" \
  --tasks ceval-valid --device cuda --batch_size 8 \
  --output_path "eval_results/$TAG" 2>&1 | tail -12
