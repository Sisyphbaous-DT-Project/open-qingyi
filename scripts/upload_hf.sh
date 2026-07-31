#!/bin/bash
GW=$(ip route show default | head -1 | cut -d' ' -f3)
echo "GW=$GW"
export HTTP_PROXY="http://$GW:7890" HTTPS_PROXY="http://$GW:7890"
cd /root/projects/qingyi-kda
.venv/bin/python -u scripts/upload_hf.py 2>&1 | tee logs/upload_hf.log
