#!/bin/sh
# robotera_vla M7 pi0.5 训练入口 (droboticflow custom-container 协议)
# 端口契约 v1.1.0: assets(第一输入)连归一化统计; 其余一切已连线端口均为数据集
# (端口名无关, 数量不限); 无 dataset 连线即失败, 不做任何回退
# 用法: sh /app/entry/train.sh   (平台通过环境变量注入全部输入)
set -eu

echo "[$(date -u +%FT%TZ)] node=${NODE_LABEL:-?} run=${RUN_ID:-?} attempt=${RUN_ATTEMPT:-1}"

: "${OUTPUT_DIR:?OUTPUT_DIR not set}"
mkdir -p "$OUTPUT_DIR"

# ---- 运行时环境 (uid 10001 无 HOME 写权限, JAX/pinocchio 需要) ----
export HOME=/tmp/home
mkdir -p "$HOME"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-/app/openpi_data}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}
export CONFIG_NAME=${CONFIG_NAME:-pi05_M7_pp_opensource}
EXP_NAME=${EXP_NAME:-${RUN_ID:-robotera-train}}

# ---- 薄拷贝运行时工作树: 符号链接只读源码, 可写目录承接产物 ----
WORK="$OUTPUT_DIR/work"
rm -rf "$WORK"
mkdir -p "$WORK"
# PVC 网络文件系统不支持批量 symlink, 源码仅 ~40MB 改实体拷贝
cp -r /app/src/. "$WORK"/
# 运行时覆盖容器内 data_config 副本 (项目源码不动)
rm -f "$WORK/training/configs/data_config_260322.yml"

# ---- 输入解析: dataset 端口集合 + assets 端口 ----
ASSETS_READY=0
PREP_OUT=$(python3 /app/adapters/prepare_inputs.py \
    --workdir "$WORK" \
    --config-name "$CONFIG_NAME" \
    --input-artifacts-path "${INPUT_ARTIFACTS_PATH:-/nonexistent}" \
    --input-bindings-path "${INPUT_BINDINGS_PATH:-/nonexistent}" \
    --dataset-ports-file "$OUTPUT_DIR/.dataset_ports") || exit 1
echo "$PREP_OUT"
case "$PREP_OUT" in *ASSETS_READY=1*) ASSETS_READY=1 ;; esac

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

if [ "$ASSETS_READY" -ne 1 ]; then
    if [ "${AUTO_NORM_STATS:-1}" = "1" ]; then
        echo "[train] norm_stats.json missing; auto-computing (SAMPLE_RATIO=${SAMPLE_RATIO:-0.1})"
        (cd "$WORK" && PYTHONPATH="$WORK" python3 scripts/compute_norm_stats.py \
            --config-name "$CONFIG_NAME" \
            --sample-ratio "${SAMPLE_RATIO:-0.1}" \
            --num-workers "${NUM_WORKERS:-16}")
    else
        echo "ERROR: norm_stats.json not found and AUTO_NORM_STATS=0" >&2
        exit 1
    fi
fi

# ---- 训练参数: 默认值 = 原版 TrainConfig 默认, 全部可用超参/透传覆盖 ----
FSDP_DEVICES=${FSDP_DEVICES:-${GPU_LIMIT:-8}}
# checkpoint 写容器本地盘(节点 ephemeral ~500G): 单份含 train_state 约 50G,
# 双份峰值 100G 会挤爆 200G 的 workspace PVC; 交付物(params+assets ~13G)由
# finalize 拷回 OUTPUT_DIR。需要跨 pod 重试续训时可将 CKPT_DIR 指向 PVC 路径。
CKPT_DIR=${CKPT_DIR:-/tmp/ckpt}
TRAIN_ARGS="--exp-name $EXP_NAME"
TRAIN_ARGS="$TRAIN_ARGS --checkpoint-base-dir $CKPT_DIR"
TRAIN_ARGS="$TRAIN_ARGS --num-train-steps ${NUM_TRAIN_STEPS:-300000}"
TRAIN_ARGS="$TRAIN_ARGS --batch-size ${BATCH_SIZE:-64}"
TRAIN_ARGS="$TRAIN_ARGS --fsdp-devices $FSDP_DEVICES"
TRAIN_ARGS="$TRAIN_ARGS --seed ${SEED:-42}"
TRAIN_ARGS="$TRAIN_ARGS --num-workers ${NUM_WORKERS:-16}"
TRAIN_ARGS="$TRAIN_ARGS --log-interval ${LOG_INTERVAL:-100}"
TRAIN_ARGS="$TRAIN_ARGS --save-interval ${SAVE_INTERVAL:-10000}"
TRAIN_ARGS="$TRAIN_ARGS --keep-period ${KEEP_PERIOD:-10000}"
TRAIN_ARGS="$TRAIN_ARGS --ema-decay ${EMA_DECAY:-0.99}"
TRAIN_ARGS="$TRAIN_ARGS --lr-schedule.warmup-steps ${LR_WARMUP_STEPS:-1000}"
TRAIN_ARGS="$TRAIN_ARGS --lr-schedule.peak-lr ${LR_PEAK_LR:-2.5e-5}"
TRAIN_ARGS="$TRAIN_ARGS --lr-schedule.decay-steps ${LR_DECAY_STEPS:-30000}"
TRAIN_ARGS="$TRAIN_ARGS --lr-schedule.decay-lr ${LR_DECAY_LR:-2.5e-6}"
TRAIN_ARGS="$TRAIN_ARGS --weight-loader.params-path ${BASE_WEIGHT_PATH:-/app/openpi_data/openpi-assets/checkpoints/pi05_base/params}"
if [ "${WANDB_ENABLED:-false}" = "true" ] || [ "${WANDB_ENABLED:-false}" = "1" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --wandb-enabled"
else
    TRAIN_ARGS="$TRAIN_ARGS --no-wandb-enabled"
fi
# 幂等: 同 run 内 pod 重试 (RUN_ATTEMPT>1, workspace 复用) 自动续训
if [ "${RUN_ATTEMPT:-1}" -gt 1 ]; then
    TRAIN_ARGS="$TRAIN_ARGS --resume"
else
    TRAIN_ARGS="$TRAIN_ARGS --overwrite"
fi

echo "[train] python scripts/train.py $CONFIG_NAME $TRAIN_ARGS ${EXTRA_TRAIN_ARGS:-}"
cd "$WORK"
PYTHONPATH="$WORK" python3 scripts/train.py "$CONFIG_NAME" $TRAIN_ARGS ${EXTRA_TRAIN_ARGS:-}

# ---- 收尾协议 ----
python3 /app/adapters/finalize.py train \
    --workdir "$WORK" \
    --ckpt-root "$CKPT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --config-name "$CONFIG_NAME" \
    --exp-name "$EXP_NAME"
