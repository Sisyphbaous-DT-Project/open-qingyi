#!/usr/bin/env bash
# 公共环境：代理（动态获取 WSL 网关 IP）+ uv 路径
GW=$(ip route | awk '/^default/{print $3}')
export HTTP_PROXY="http://${GW}:7890"
export HTTPS_PROXY="http://${GW}:7890"
export PATH="$HOME/.local/bin:$PATH"
