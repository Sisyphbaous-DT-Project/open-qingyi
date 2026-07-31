#!/bin/bash
# 监视 v2 蒸馏检查点：新 step-N 出现就复制权重到队列（防轮换），逐个跑 C-Eval
ROOT=/root/autodl-tmp/qingyi-kda
CK=$ROOT/models/distill-v2-checkpoints
Q=$ROOT/models/distill-v2-eval-queue
DONE=$ROOT/logs/distill_v2_evaluated.txt
SUM=$ROOT/logs/distill_v2_eval_summary.txt
mkdir -p "$Q"
touch "$DONE" "$SUM"

while true; do
  for d in "$CK"/step-*; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    case "$name" in *.tmp) continue;; esac
    grep -qx "$name" "$DONE" && continue
    sf="$d/model.safetensors"
    [ -f "$sf" ] || continue
    s1=$(stat -c%s "$sf"); sleep 20; s2=$(stat -c%s "$sf")
    [ "$s1" = "$s2" ] || continue
    cp "$sf" "$Q/$name.safetensors"
    echo "$name" >> "$DONE"
    echo "[watcher] queued $name"
  done

  for f in "$Q"/step-*.safetensors; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .safetensors)
    L=$ROOT/models/eval-loadable-dv2-$name
    rm -rf "$L"
    cp -r "$ROOT/models/hf-skeleton" "$L"
    cp "$f" "$L/model.safetensors"
    cd "$ROOT"
    echo "[watcher] evaluating $name"
    HF_ENDPOINT=https://hf-mirror.com /root/miniconda3/bin/python -m lm_eval \
      --model hf --model_args pretrained="$L",dtype=bfloat16,trust_remote_code=True \
      --tasks ceval-valid --device cuda --batch_size 8 \
      --output_path "eval_results/dv2-$name" > "logs/eval_dv2_$name.log" 2>&1
    line=$(grep -E '\|ceval-valid' "logs/eval_dv2_$name.log" | grep 'acc ' | tail -1)
    echo "$name | $line" >> "$SUM"
    echo "[watcher] done $name : $line"
    rm -rf "$L" "$f"
  done

  if [ -f "$ROOT/logs/distill_v2.done" ]; then
    if ! ls "$Q"/step-*.safetensors >/dev/null 2>&1; then
      echo "[watcher] ALL-DONE"
      break
    fi
  fi
  sleep 60
done
