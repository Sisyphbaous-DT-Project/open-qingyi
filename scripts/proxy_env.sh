#!/bin/bash
# Source this to get proxy env pointing at the Windows host Clash proxy.
GW=$(ip route show default | head -1 | cut -d' ' -f3)
export HTTP_PROXY="http://$GW:7890"
export HTTPS_PROXY="http://$GW:7890"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
echo "proxy=$HTTP_PROXY"
