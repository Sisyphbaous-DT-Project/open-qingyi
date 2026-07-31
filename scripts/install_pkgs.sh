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

# torch: CUDA 12.x Linux wheel（PyPI 默认即 CUDA 构建）
retry uv pip install --python .venv/bin/python torch

retry uv pip install --python .venv/bin/python \
  transformers \
  "fla-core>=0.4.0" \
  bitsandbytes \
  datasets \
  accelerate \
  sentencepiece

echo "=== installed versions ==="
.venv/bin/python -m pip list 2>/dev/null | grep -Ei 'torch|transformers|fla|triton|bitsandbytes|datasets|accelerate|sentencepiece' || true
