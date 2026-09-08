#!/bin/sh
# robotera_vla M7 归一化统计入口 (droboticflow custom-container 协议)
# 端口契约 v1.1.0: 一切已连线端口均为数据集(端口名无关, 数量不限);
# 无连线即失败, 不做任何回退
set -eu

echo "[$(date -u +%FT%TZ)] node=${NODE_LABEL:-?} run=${RUN_ID:-?}"

: "${OUTPUT_DIR:?OUTPUT_DIR not set}"
mkdir -p "$OUTPUT_DIR"

export HOME=/tmp/home
mkdir -p "$HOME"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export CONFIG_NAME=${CONFIG_NAME:-pi05_M7_pp_opensource}

WORK="$OUTPUT_DIR/work"
rm -rf "$WORK"
mkdir -p "$WORK"
cp -r /app/src/. "$WORK"/
rm -f "$WORK/training/configs/data_config_260322.yml"

PREP_OUT=$(python3 /app/adapters/prepare_inputs.py \
    --workdir "$WORK" \
    --config-name "$CONFIG_NAME" \
    --input-artifacts-path "${INPUT_ARTIFACTS_PATH:-/nonexistent}" \
    --input-bindings-path "${INPUT_BINDINGS_PATH:-/nonexistent}" \
    --dataset-ports-file "$OUTPUT_DIR/.dataset_ports") || exit 1
echo "$PREP_OUT"

GEN_ARGS="--out $WORK/training/configs/data_config_260322.yml"
if grep -q . "$OUTPUT_DIR/.dataset_ports" 2>/dev/null; then
    while IFS= read -r repo; do
        [ -n "$repo" ] && GEN_ARGS="$GEN_ARGS --repo $repo"
    done < "$OUTPUT_DIR/.dataset_ports"
else
    echo "ERROR: no dataset ports wired; datasets can only enter via port wiring (目录读取 -> dataset 端口连线)" >&2
    exit 1
fi
python3 /app/adapters/gen_data_config.py $GEN_ARGS

NORM_ARGS="--config-name $CONFIG_NAME"
NORM_ARGS="$NORM_ARGS --sample-ratio ${SAMPLE_RATIO:-0.1}"
NORM_ARGS="$NORM_ARGS --num-workers ${NUM_WORKERS:-8}"
if [ -n "${MAX_FRAMES:-}" ]; then
    NORM_ARGS="$NORM_ARGS --max-frames $MAX_FRAMES"
fi

echo "[norm_stats] python scripts/compute_norm_stats.py $NORM_ARGS ${EXTRA_NORM_ARGS:-}"
cd "$WORK"
PYTHONPATH="$WORK" python3 scripts/compute_norm_stats.py $NORM_ARGS ${EXTRA_NORM_ARGS:-}

python3 /app/adapters/finalize.py norm_stats \
    --workdir "$WORK" \
    --output-dir "$OUTPUT_DIR" \
    --config-name "$CONFIG_NAME"
