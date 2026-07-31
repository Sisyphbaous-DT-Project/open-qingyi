#!/bin/bash
cd /root/projects/qingyi-kda
exec .venv/bin/python -u -m lm_eval \
  --model hf --model_args pretrained=models/distill-best-loadable,dtype=bfloat16,trust_remote_code=True \
  --tasks ceval-valid --device cuda --batch_size 8 \
  --output_path eval_results/distill-ceval >> logs/eval_distill.log 2>&1
