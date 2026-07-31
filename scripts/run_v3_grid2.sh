#!/bin/bash
# v3 衰减细化：g6/g7 两组
cd /root/projects/qingyi-kda
PY=.venv/bin/python
run() {  # name a_low a_high
  echo "=== grid $1: scale=0.02 dt=0 A=($2,$3) ==="
  $PY scripts/build_hybrid_v3.py --gate-scale 0.02 --dt-bias 0 --a-low "$2" --a-high "$3" \
      --out "models/v3grid-$1" > "logs/v3grid-$1.log" 2>&1 \
    && $PY scripts/eval_heldout.py "models/v3grid-$1" --tag "v3grid-$1" >> "logs/v3grid-$1.log" 2>&1
  grep overall "logs/v3grid-$1.log" | tail -1
}
run g6 0.03 0.3
run g7 0.005 0.05
echo REFINE_DONE
