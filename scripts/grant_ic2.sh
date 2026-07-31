#!/usr/bin/env bash
set -u
source /root/projects/qingyi-kda/scripts/env.sh
TOKEN=$(cat ~/.cache/huggingface/token)
for endpoint in ask-access accept; do
  code=$(curl -s -o "/tmp/grant_$endpoint.json" -w '%{http_code}' -X POST \
    -H "Authorization: Bearer $TOKEN" \
    "https://huggingface.co/api/datasets/BAAI/IndustryCorpus2/$endpoint")
  echo "$endpoint -> HTTP $code : $(head -c 200 /tmp/grant_$endpoint.json 2>/dev/null)"
done
