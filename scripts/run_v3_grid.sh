#!/bin/bash
# v3 门控初始化小网格：每组 构建(存 models/v3grid-*) + 新尺评测，日志 logs/v3grid-*.log
cd /root/projects/qingyi-kda
PY=.venv/bin/python
run() {  # name gate_scale dt_bias a_low a_high
  echo "=== grid $1: scale=$2 dt=$3 A=($4,$5) ==="
  $PY scripts/build_hybrid_v3.py --gate-scale "$2" --dt-bias "$3" --a-low "$4" --a-high "$5" \
      --out "models/v3grid-$1" > "logs/v3grid-$1.log" 2>&1 \
    && $PY scripts/eval_heldout.py "models/v3grid-$1" --tag "v3grid-$1" >> "logs/v3grid-$1.log" 2>&1
  grep -E "overall|saved" "logs/v3grid-$1.log" | tail -2
}
run g2 0.02 0    0.01 0.1
run g3 0.02 2    0.1  1
run g4 0.2  0    0.01 0.1
run g5 0.02 -2   0.01 0.1
echo GRID_DONE
