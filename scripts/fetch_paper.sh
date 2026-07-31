#!/usr/bin/env bash
set -euo pipefail
source /root/projects/qingyi-kda/scripts/env.sh
cd /root/projects/qingyi-kda
GW=$(ip route | awk '/^default/{print $3}')
.venv/bin/uv pip install --python .venv/bin/python pypdf 2>/dev/null || uv pip install --python .venv/bin/python pypdf
curl -sL --proxy "http://${GW}:7890" "https://arxiv.org/pdf/2510.26692" -o data/kimi_linear.pdf
ls -lh data/kimi_linear.pdf
.venv/bin/python - <<'EOF'
from pypdf import PdfReader
r = PdfReader("data/kimi_linear.pdf")
print("pages:", len(r.pages))
text = "\n\n".join((p.extract_text() or "") for p in r.pages)
open("data/kimi_linear.txt", "w").write(text)
print("chars:", len(text))
EOF
