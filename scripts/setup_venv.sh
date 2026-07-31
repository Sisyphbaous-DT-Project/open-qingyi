#!/usr/bin/env bash
set -euo pipefail
source /root/projects/qingyi-kda/scripts/env.sh
cd /root/projects/qingyi-kda
uv python install 3.12
uv venv --python 3.12 .venv
.venv/bin/python --version
