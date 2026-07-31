#!/usr/bin/env bash
set -euo pipefail
source /root/projects/qingyi-kda/scripts/env.sh

# HF token auth (token passed via env HF_TOKEN_VALUE, never written to project files)
mkdir -p ~/.cache/huggingface
printf '%s' "$HF_TOKEN_VALUE" > ~/.cache/huggingface/token
chmod 600 ~/.cache/huggingface/token

# Verify: datasets API for BAAI/IndustryCorpus2
code=$(curl -s -o /tmp/ic2.json -w '%{http_code}' \
  -H "Authorization: Bearer $HF_TOKEN_VALUE" \
  "https://huggingface.co/api/datasets/BAAI/IndustryCorpus2")
echo "IndustryCorpus2 API status: $code"
/root/projects/qingyi-kda/.venv/bin/python - <<'EOF'
import json
d = json.load(open("/tmp/ic2.json"))
print("gated:", d.get("gated"), "| private:", d.get("private"))
siblings = [f["rfilename"] for f in d.get("siblings", [])]
print("n files:", len(siblings))
print([s for s in siblings[:10]])
EOF
