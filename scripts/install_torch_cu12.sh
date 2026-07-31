#!/usr/bin/env bash
set -euo pipefail
source /root/projects/qingyi-kda/scripts/env.sh
cd /root/projects/qingyi-kda

retry() {
  local n=0
  until "$@"; do
    n=$((n+1))
    if [ "$n" -ge 8 ]; then
      echo "FAILED after $n attempts: $*" >&2
      return 1
    fi
    echo "attempt $n failed, retrying in 10s..." >&2
    sleep 10
  done
}

# 用 PyTorch 官方 cu128 索引重装 torch（CUDA 12.8 wheel）
# cu128 索引上最新为 torch 2.11.0+cu128；显式 pin 版本避免 uv 认为已满足
retry uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.11.0"

.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
