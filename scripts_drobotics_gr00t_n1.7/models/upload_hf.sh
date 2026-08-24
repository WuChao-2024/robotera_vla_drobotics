#!/usr/bin/env bash
# 上传本目录(models/)到 HuggingFace,仓库类型: 模型权重
set -euo pipefail

# 1. token(替换成自己的)
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx

# 2. 目标仓库(替换成自己的 用户名/仓库名)
REPO="WuChao-Cauchy/gr00t_n1.7_m7_pick_place_weights"

# 本机访问 HF 需走代理
export https_proxy=http://192.168.16.68:18000
export http_proxy=http://192.168.16.68:18000

cd "$(dirname "$0")"
hf upload "$REPO" . . --repo-type model
