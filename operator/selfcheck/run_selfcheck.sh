#!/bin/bash
# A9 出厂自检: 以 uid 10001 运行镜像, 逐项验证 droboticflow 算子契约。
# 数据集通过伪造的 INPUT_BINDINGS 模拟端口连线注入(与平台运行时同构, 不依赖超参回退)。
# 用法: IMAGE=<镜像tag> DATASET=<本机单个 LeRobot collection 根目录(含 meta/+data/)> bash run_selfcheck.sh
set -euo pipefail

IMAGE="${IMAGE:?set IMAGE=<tag>}"
DATASET="${DATASET:?set DATASET=<local lerobot collection root>}"
SCRATCH="$(mktemp -d /tmp/rv_selfcheck.XXXX)"
KEEP_OUT="$SCRATCH/out"
trap 'echo "selfcheck scratch: $SCRATCH"' EXIT

run_case() {
    local name="$1"; shift
    echo "===== CASE: $name ====="
    if "$@"; then
        echo "----- CASE $name: exit 0"
    else
        echo "----- CASE $name: exit $?"
    fi
}

# 公共环境 (平台保证注入的变量)
common_env=(
    -e OUTPUT_DIR=/out
    -e RESULT_OUTPUT_PATH=/out/result.json
    -e STAGE_MANIFEST_PATH=/out/stage-manifest.json
    -e WORKFLOW_ID=wf-test -e RUN_ID=run-test -e NODE_ID=node-test -e NODE_LABEL=selfcheck
    -e INPUT_ARTIFACTS_PATH=/out/input-artifacts.json
    -e INPUT_BINDINGS_PATH=/out/input-bindings.json
    -v "$KEEP_OUT:/out"
    -v "$DATASET:$DATASET:ro"
)

# 模拟端口连线: 每行参数为一个 (targetPortId, path), 写成 bindings JSON
wire_bindings() {
    python3 - "$KEEP_OUT/input-bindings.json" "$@" <<'PYEOF'
import json, sys, pathlib
out, pairs = sys.argv[1], sys.argv[2:]
bindings = []
for i in range(0, len(pairs), 2):
    bindings.append({"targetPortId": pairs[i], "artifacts": [{"path": pairs[i+1]}]})
pathlib.Path(out).write_text(json.dumps(bindings))
PYEOF
}

# CASE 1: norm_stats 小采样全流程 (uid 10001, 声明端口 dataset)
rm -rf "$KEEP_OUT"; mkdir -p "$KEEP_OUT"; echo '[]' > "$KEEP_OUT/input-artifacts.json"
wire_bindings dataset "$DATASET"
run_case "norm_stats_uid10001" docker run --rm -u 10001:10001 "${common_env[@]}" \
    -e SAMPLE_RATIO=0.002 -e NUM_WORKERS=2 -e MAX_FRAMES=64 \
    "$IMAGE" sh /app/entry/norm_stats.sh

echo "--- artifacts of case1:"
[ -f "$KEEP_OUT/result.json" ] && cat "$KEEP_OUT/result.json"
[ -f "$KEEP_OUT/work/assets/pi05_M7_pp_opensource/norm_stats.json" ] && echo "norm_stats.json OK"

# CASE 2: train 快速链路 (CPU 强制, 只验证 参数拼装/数据集/缓存命中/收尾; GPU 训练在平台验证)
rm -rf "$KEEP_OUT"; mkdir -p "$KEEP_OUT"; echo '[]' > "$KEEP_OUT/input-artifacts.json"
wire_bindings dataset "$DATASET"
run_case "train_cpu_smoke" docker run --rm -u 10001:10001 "${common_env[@]}" \
    -e JAX_PLATFORMS=name -e NUM_TRAIN_STEPS=1 -e BATCH_SIZE=1 \
    -e FSDP_DEVICES=1 -e NUM_WORKERS=2 -e SAVE_INTERVAL=1 -e SAMPLE_RATIO=0.002 \
    "$IMAGE" sh /app/entry/train.sh

# CASE 3: 无任何 dataset 端口连线, 必须非 0 退出且报错可读
rm -rf "$KEEP_OUT"; mkdir -p "$KEEP_OUT"; echo '[]' > "$KEEP_OUT/input-artifacts.json"; echo '[]' > "$KEEP_OUT/input-bindings.json"
run_case "no_dataset_ports" docker run --rm -u 10001:10001 "${common_env[@]}" \
    "$IMAGE" sh /app/entry/norm_stats.sh

# CASE 4: UUID 端口名也必须被当成数据集 (v1.1.0 契约: 端口名无关)
rm -rf "$KEEP_OUT"; mkdir -p "$KEEP_OUT"; echo '[]' > "$KEEP_OUT/input-artifacts.json"
wire_bindings "3f2a1b8c-9d4e-4f6a-b1c2-0123456789ab" "$DATASET"
run_case "uuid_dataset_port" docker run --rm -u 10001:10001 "${common_env[@]}" \
    -e SAMPLE_RATIO=0.002 -e NUM_WORKERS=2 -e MAX_FRAMES=64 \
    "$IMAGE" sh /app/entry/norm_stats.sh

# CASE 5: assets 端口连了不含 norm_stats.json 的目录, 必须非 0 退出且报错可读
rm -rf "$KEEP_OUT"; mkdir -p "$KEEP_OUT"; echo '[]' > "$KEEP_OUT/input-artifacts.json"
wire_bindings dataset "$DATASET" assets "$DATASET"
run_case "assets_wrong_content" docker run --rm -u 10001:10001 "${common_env[@]}" \
    -e JAX_PLATFORMS=name -e NUM_TRAIN_STEPS=1 -e BATCH_SIZE=1 -e FSDP_DEVICES=1 \
    "$IMAGE" sh /app/entry/train.sh

# CASE 6: RUN_ATTEMPT=2 时 train 命令应包含 --resume (从日志 grep)
echo "===== CASE: resume_flag ====="
rm -rf "$KEEP_OUT"; mkdir -p "$KEEP_OUT"; echo '[]' > "$KEEP_OUT/input-artifacts.json"
wire_bindings dataset "$DATASET"
docker run --rm -u 10001:10001 "${common_env[@]}" \
    -e RUN_ATTEMPT=2 -e JAX_PLATFORMS=name \
    -e NUM_TRAIN_STEPS=1 -e BATCH_SIZE=1 -e FSDP_DEVICES=1 -e NUM_WORKERS=2 -e SAMPLE_RATIO=0.002 \
    "$IMAGE" sh /app/entry/train.sh 2>&1 | grep -E "\-\-resume|\[train\] python" || echo "grep found nothing"

echo "===== selfcheck done ====="
