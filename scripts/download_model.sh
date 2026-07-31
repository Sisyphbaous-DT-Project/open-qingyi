#!/usr/bin/env bash
set -euo pipefail
source /root/projects/qingyi-kda/scripts/env.sh
cd /root/projects/qingyi-kda
.venv/bin/hf download Qwen/Qwen3-0.6B-Base \
  --local-dir /root/projects/qingyi-kda/models/Qwen3-0.6B-Base
echo "=== downloaded files ==="
ls -lh /root/projects/qingyi-kda/models/Qwen3-0.6B-Base
