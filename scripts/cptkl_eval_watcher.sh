#!/bin/bash
# 监视 CPT+KL 检查点：新 step-N 出现就复制权重到队列（防轮换），逐个跑 C-Eval
ROOT=/root/autodl-tmp/qingyi-kda
CK=$ROOT/models/cptkl-checkpoints
Q=$ROOT/models/cptkl-eval-queue
DONE=$ROOT/logs/cptkl_evaluated.txt
SUM=$ROOT/logs/cptkl_eval_summary.txt
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
    L=$ROOT/models/eval-loadable-cptkl-$name
    rm -rf "$L"
    cp -r "$ROOT/models/hf-skeleton" "$L"
    cp "$f" "$L/model.safetensors"
    cd "$ROOT"
    echo "[watcher] evaluating $name"
    HF_ENDPOINT=https://hf-mirror.com /root/miniconda3/bin/python -m lm_eval \
      --model hf --model_args pretrained="$L",dtype=bfloat16,trust_remote_code=True \
      --tasks ceval-valid --device cuda --batch_size 8 \
      --output_path "eval_results/cptkl-$name" > "logs/eval_cptkl_$name.log" 2>&1
    line=$(grep -E '\|ceval-valid' "logs/eval_cptkl_$name.log" | grep 'acc ' | tail -1)
    echo "$name | $line" >> "$SUM"
    echo "[watcher] done $name : $line"
    rm -rf "$L" "$f"
  done

  if [ -f "$ROOT/logs/cptkl.done" ]; then
    if ! ls "$Q"/step-*.safetensors >/dev/null 2>&1; then
      echo "[watcher] ALL-DONE"
      break
    fi
  fi
  sleep 60
done
