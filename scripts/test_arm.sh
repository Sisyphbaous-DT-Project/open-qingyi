#!/bin/bash
# 参数化臂验收：C-Eval + 生成面板（单模型，不跑 best-ce，省时省盘）
# --log_samples 保留逐题结果，供 A/B 同题配对比较
# 成功时写 /root/autodl-tmp/<TAG>.{ceval,gen}.done 标记；PID 写 .pid 文件
# Usage: test_arm.sh <TAG> <CKPT_DIR>   e.g. test_arm.sh arma models/arm-a-checkpoints/step-7500
set -e
TAG=$1
SRC=$2
cd /root/autodl-tmp/qingyi-kda

rm -rf "models/eval-loadable-$TAG" /root/autodl-tmp/$TAG.ceval.done /root/autodl-tmp/$TAG.gen.done
mkdir -p "models/eval-loadable-$TAG"
cp models/hf-skeleton/* "models/eval-loadable-$TAG"/
sed -i 's/"head_dim": 64/"head_dim": 128/' "models/eval-loadable-$TAG/config.json"
cp "$SRC/model.safetensors" "$SRC/layout.json" "models/eval-loadable-$TAG"/

export HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1

nohup bash -c "/root/miniconda3/bin/python -m lm_eval --model hf \
  --model_args pretrained=/root/autodl-tmp/qingyi-kda/models/eval-loadable-$TAG,dtype=bfloat16,trust_remote_code=True \
  --tasks ceval-valid --device cuda --batch_size 8 --log_samples \
  --output_path /root/autodl-tmp/qingyi-kda/eval_results/$TAG-ceval \
  && touch /root/autodl-tmp/$TAG.ceval.done" \
  > /root/autodl-tmp/$TAG-ceval.log 2>&1 &
echo $! > /root/autodl-tmp/$TAG.ceval.pid
echo CEVAL_${TAG}_LAUNCHED

nohup bash -c "/root/miniconda3/bin/python scripts/generate_sample.py \
  '$SRC' --tokens 80 \
  && touch /root/autodl-tmp/$TAG.gen.done" \
  > /root/autodl-tmp/$TAG-gen.txt 2>&1 &
echo $! > /root/autodl-tmp/$TAG.gen.pid
echo GEN_${TAG}_LAUNCHED
