#!/bin/bash
# 组装 staging context 并构建 robotera-vla-train 镜像, 推送 CCR。
# 项目源码零修改: staging 目录在 /tmp 组装, 构建完即删。
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OPERATOR_DIR="$PROJ_ROOT/operator"
WEIGHTS_SRC="${WEIGHTS_SRC:-/home/chao01.wu/DeployAnything/DataPipeline/OpenPi_Dev/weights_cache/openpi_weights}"
CCR_REGISTRY="${CCR_REGISTRY:-ccr-29eug8s3-pub.cnc.bj.baidubce.com/dataflow-poc/custom-operators/robotera-vla-train}"
STAGE="${STAGE:-/tmp/rv_stage}"

command -v docker >/dev/null || { echo "docker not found"; exit 1; }
[ -d "$WEIGHTS_SRC/pi05_base/params" ] || { echo "missing $WEIGHTS_SRC/pi05_base/params"; exit 1; }
[ -f "$WEIGHTS_SRC/big_vision/paligemma_tokenizer.model" ] || { echo "missing paligemma tokenizer"; exit 1; }

echo "[build] assembling staging context at $STAGE"
rm -rf "$STAGE"
mkdir -p "$STAGE"

# 项目源码 (排除与训练镜像无关的目录)
rsync -a \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='docs' \
    --exclude='scripts_drobotics' \
    --exclude='scripts_drobotics_gr00t_n1.7' \
    --exclude='operator' \
    "$PROJ_ROOT/" "$STAGE/src/"

# operator 三件套
cp -r "$OPERATOR_DIR/entry" "$STAGE/entry"
cp -r "$OPERATOR_DIR/adapters" "$STAGE/adapters"
find "$STAGE/entry" "$STAGE/adapters" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
cp "$OPERATOR_DIR/Dockerfile" "$STAGE/Dockerfile"

# 预置权重缓存: pi05_base 基座 12G + tokenizer (镜像内置, 离线运行)
mkdir -p "$STAGE/weights"
cp -r "$WEIGHTS_SRC/pi05_base" "$STAGE/weights/pi05_base"
mkdir -p "$STAGE/weights/big_vision"
cp "$WEIGHTS_SRC/big_vision/paligemma_tokenizer.model" "$STAGE/weights/big_vision/"

du -sh "$STAGE"

export https_proxy=http://192.168.16.68:18000
export http_proxy=http://192.168.16.68:18000
docker buildx use proxybuilder68

TAG="$(date +%Y%m%d_%H%M%S)"
echo "$TAG" > /tmp/rv_cur_tag
IMAGE="$CCR_REGISTRY:$TAG"
echo "[build] image = $IMAGE"

setsid nohup docker buildx build --builder proxybuilder68 \
    -f "$STAGE/Dockerfile" \
    --build-arg http_proxy="$http_proxy" \
    --build-arg https_proxy="$https_proxy" \
    --build-arg no_proxy="localhost,127.0.0.1" \
    -t "$IMAGE" \
    --push \
    "$STAGE" > /tmp/rv_build.log 2>&1 &
echo "[build] background build started, log: /tmp/rv_build.log"
echo "[build] monitor: tail -f /tmp/rv_build.log"
