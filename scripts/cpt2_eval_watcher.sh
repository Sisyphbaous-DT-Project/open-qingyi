#!/bin/bash
# 监视 cpt2 检查点：出现新 step-N 就把权重复制到队列（防轮换），逐个跑 C-Eval
ROOT=/root/autodl-tmp/qingyi-kda
CK=$ROOT/models/cpt2-checkpoints
Q=$ROOT/models/cpt2-eval-queue
DONE=$ROOT/logs/cpt2_evaluated.txt
SUM=$ROOT/logs/cpt2_eval_summary.txt
mkdir -p "$Q"
touch "$DONE" "$SUM"

while true; do
  # 1) 发现新检查点 -> 复制权重到队列
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

  # 2) 逐个评测队列
  for f in "$Q"/step-*.safetensors; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .safetensors)
    L=$ROOT/models/eval-loadable-cpt2-$name
    rm -rf "$L"
    cp -r "$ROOT/models/hf-skeleton" "$L"
    cp "$f" "$L/model.safetensors"
    cd "$ROOT"
    echo "[watcher] evaluating $name"
    HF_ENDPOINT=https://hf-mirror.com /root/miniconda3/bin/python -m lm_eval \
      --model hf --model_args pretrained="$L",dtype=bfloat16,trust_remote_code=True \
      --tasks ceval-valid --device cuda --batch_size 8 \
      --output_path "eval_results/cpt2-$name" > "logs/eval_cpt2_$name.log" 2>&1
    line=$(grep -E '\|ceval-valid' "logs/eval_cpt2_$name.log" | grep 'acc ' | tail -1)
    echo "$name | $line" >> "$SUM"
    echo "[watcher] done $name : $line"
    rm -rf "$L" "$f"
  done

  # 3) 训练结束且队列清空 -> 退出
  if [ -f "$ROOT/logs/cpt2.done" ]; then
    if ! ls "$Q"/step-*.safetensors >/dev/null 2>&1; then
      echo "[watcher] ALL-DONE"
      break
    fi
  fi
  sleep 60
done
