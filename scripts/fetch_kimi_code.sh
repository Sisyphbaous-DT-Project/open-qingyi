#!/usr/bin/env bash
set -euo pipefail
source /root/projects/qingyi-kda/scripts/env.sh
cd /root/projects/qingyi-kda/data
for f in modeling_kimi.py configuration_kimi.py; do
  curl -sL "https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/raw/main/$f" -o "$f"
done
wc -l modeling_kimi.py configuration_kimi.py
