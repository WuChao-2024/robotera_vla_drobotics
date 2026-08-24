#!/bin/bash
# 启动 gr00t_release 板端 BPU 推理服务(最终版: action 塔融合 + 全链路优化, 纯计算稳态 97ms)
# 模型资产在 bpu_model/v1/。用法: bash run_serve.sh [端口]   默认 5558; 端口被占时换一个
cd "$(dirname "$0")"
export GR00T_M7_URDF="$(pwd)/bpu_model/v1/l3_4.urdf"
PORT=${1:-5558}
setsid nohup python3 serve_m7_eval_bpu_fused_opt.py \
  --vbm bpu_model/v1/vision.hbm --lbm bpu_model/v1/language.hbm --abm bpu_model/v1/action_fused.hbm \
  --stats bpu_model/v1/statistics.json --cosmos bpu_model/v1/cosmos --embed bpu_model/v1/embed_tokens.npy \
  --port $PORT > serve.log 2>&1 &
echo "serve pid=$! port=$PORT log=$(pwd)/serve.log"
