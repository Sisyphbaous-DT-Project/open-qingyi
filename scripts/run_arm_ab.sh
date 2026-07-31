#!/bin/bash
# 三臂对照之 A/B 臂：同起点（kept step-7000）、同数据游标、各跑 500 步。
# A = T=1 原配方延续（lr_at 已钳制 t<=1，全程恒定 min_lr 2.5e-6）
# B = T=2 温度软化（唯一变量 temperature，需 --allow-hparam-override）
# 两臂 resume 同一 checkpoint：模型/优化器/数据游标/RNG 完全一致，严格同起点。
# 在云项目根 /root/autodl-tmp/qingyi-kda 下执行。
set -e
cd /root/autodl-tmp/qingyi-kda
PY=/root/miniconda3/bin/python
HASH=2d7b0b7f5ff05fa8477a08d030c20867d70ac4b4ae9d2d673bebe77f2210007c
CKPT=models/kept-checkpoints/step-7000

fail () {
  echo "FATAL: 参数缺失或环境不满足: $1" >&2
  exit 1
}

preflight () {
  # 起点完整性
  for f in model.safetensors layout.json trainer_state.json optimizer.pt master.pt rng.pt; do
    [ -f "$CKPT/$f" ] || fail "$CKPT/$f missing"
  done
  STEP=$($PY -c "import json;print(json.load(open('$CKPT/trainer_state.json'))['step'])")
  [ "$STEP" = "7000" ] || fail "trainer_state.step=$STEP != 7000"
  # 输出目录必须为空（防旧终点检查点撑爆原子保存）
  [ ! -e models/arm-a-checkpoints ] || fail "models/arm-a-checkpoints already exists"
  [ ! -e models/arm-b-checkpoints ] || fail "models/arm-b-checkpoints already exists"
  # 磁盘：两臂各约10G + 原子保存瞬时峰值，要求至少 22G 可用
  FREE_G=$(df -BG /root/autodl-tmp | tail -1 | tr -s ' ' | cut -d' ' -f4 | tr -d 'G')
  [ "$FREE_G" -ge 22 ] || fail "disk free ${FREE_G}G < 22G"
  echo "PREFLIGHT OK (step=$STEP, free=${FREE_G}G)"
}

kill_eval () {
  # 先 TERM，等 5 秒，仍存活再 KILL
  local TAG=$1 PID
  for KIND in ceval gen; do
    PID=$(cat /root/autodl-tmp/$TAG.$KIND.pid 2>/dev/null || true)
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
      kill -TERM "$PID" 2>/dev/null || true
    fi
  done
  sleep 5
  for KIND in ceval gen; do
    PID=$(cat /root/autodl-tmp/$TAG.$KIND.pid 2>/dev/null || true)
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
      kill -KILL "$PID" 2>/dev/null || true
      echo "killed lingering $TAG.$KIND (pid $PID)" >&2
    fi
  done
}

wait_eval () {
  # 成功判据：两个 .done 标记都存在（命令 exit 0 才会写）。
  # 进程全退但无 .done = 失败；40 分钟超时同样失败；失败先杀残留进程再退出。
  local TAG=$1 ELAPSED=0
  local CPID GPID C_OK G_OK
  CPID=$(cat /root/autodl-tmp/$TAG.ceval.pid)
  GPID=$(cat /root/autodl-tmp/$TAG.gen.pid)
  while true; do
    C_OK=0; G_OK=0
    [ -f /root/autodl-tmp/$TAG.ceval.done ] && C_OK=1
    [ -f /root/autodl-tmp/$TAG.gen.done ] && G_OK=1
    if [ "$C_OK" = "1" ] && [ "$G_OK" = "1" ]; then
      break
    fi
    if ! kill -0 "$CPID" 2>/dev/null && ! kill -0 "$GPID" 2>/dev/null; then
      echo "EVAL $TAG FAILED (processes exited without .done markers)" >&2
      tail -20 /root/autodl-tmp/$TAG-ceval.log >&2 || true
      tail -20 /root/autodl-tmp/$TAG-gen.txt >&2 || true
      exit 1
    fi
    ELAPSED=$((ELAPSED + 30))
    if [ "$ELAPSED" -ge 2400 ]; then
      echo "EVAL $TAG TIMEOUT after 40min, killing eval processes" >&2
      kill_eval "$TAG"
      tail -20 /root/autodl-tmp/$TAG-ceval.log >&2 || true
      tail -20 /root/autodl-tmp/$TAG-gen.txt >&2 || true
      exit 1
    fi
    sleep 30
  done
  rm -rf models/eval-loadable-$TAG
  echo "EVAL $TAG COLLECTED"
}

preflight

echo "=== ARM A: T=1 continue (min_lr constant) ==="
$PY scripts/kd_e2e.py \
  --resume $CKPT \
  --run-until 7500 \
  --eval-every 100 --save-every 500 --log-every 25 \
  --micro-batch 2 --grad-accum 2 \
  --expect-init-hash $HASH \
  --out models/arm-a-checkpoints \
  > /root/autodl-tmp/arm-a.log 2>&1
echo "ARM A TRAIN DONE"

bash scripts/test_arm.sh arma models/arm-a-checkpoints/step-7500
wait_eval arma
echo "=== ARM A FULLY DONE ==="

echo "=== ARM B: T=2 temperature ==="
$PY scripts/kd_e2e.py \
  --resume $CKPT \
  --run-until 7500 \
  --eval-every 100 --save-every 500 --log-every 25 \
  --micro-batch 2 --grad-accum 2 \
  --temperature 2.0 --allow-hparam-override \
  --expect-init-hash $HASH \
  --out models/arm-b-checkpoints \
  > /root/autodl-tmp/arm-b.log 2>&1
echo "ARM B TRAIN DONE"

bash scripts/test_arm.sh armb models/arm-b-checkpoints/step-7500
wait_eval armb
echo "=== ARM B FULLY DONE ==="
echo "ALL ARMS DONE"
