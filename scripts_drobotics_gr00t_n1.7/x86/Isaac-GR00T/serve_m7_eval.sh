#!/usr/bin/env bash
# M7 评测专用推理 server (适配 star1-benchmark 的裸 ZMQ 协议)。
# 默认 GPU2 + port 5556 (避开 GPU0=openpi, GPU1=serve_m7.sh:5555)。
# 用法: bash serve_m7_eval.sh [--device cuda:3 --port 5556 ...]
set -euo pipefail
export PATH=/home/chao01.wu/miniconda3/envs/gr00t/bin:$PATH
export HF_HOME=/mnt/data/chao01.wu/hf_cache
export HF_HUB_OFFLINE=1
export GR00T_COSMOS_PATH=/home/chao01.wu/DeployAnything/GR00T_Dev/weights/Cosmos-Reason2-2B
# torchcodec/ffmpeg 不需要 (评测端读视频), 但保留 gr00t lib 路径以防万一
export LD_LIBRARY_PATH=/home/chao01.wu/miniconda3/envs/gr00t/lib:${LD_LIBRARY_PATH:-}
export https_proxy=http://192.168.16.68:18000
export http_proxy=http://192.168.16.68:18000
exec /home/chao01.wu/miniconda3/envs/gr00t/bin/python \
  /home/chao01.wu/DeployAnything/GR00T_Dev/Isaac-GR00T/serve_m7_eval.py "$@"
