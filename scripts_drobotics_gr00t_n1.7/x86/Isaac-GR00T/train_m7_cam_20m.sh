#!/usr/bin/env bash
# M7 双臂人形 相机系(camera_frame) 20M 样本长训练 (4 x RTX 5090D, GPU0-3)
# 超参全部用 GR00T 官方 FinetuneConfig 默认 (bs64/lr1e-4/wd1e-5/warmup0.05/accum1)
# max_steps=312500 = 20M样本/64; save_steps=50000 密集中途评测
# OOM 退路: --global-batch-size 32 --gradient-accumulation-steps 2 (每卡8, effective 64)
set -euo pipefail

export PATH=/home/chao01.wu/miniconda3/envs/gr00t/bin:$PATH
export HF_HOME=/mnt/data/chao01.wu/hf_cache
export GR00T_COSMOS_PATH=/home/chao01.wu/DeployAnything/GR00T_Dev/weights/Cosmos-Reason2-2B
# camera_frame FK 需要 URDF (camera_frame.py 默认值即此路径, 显式 export 防 worker 找不到)
export GR00T_M7_URDF=/home/chao01.wu/DeployAnything/GR00T_Dev/robotera_vla_drobotics/l3_4.urdf
# M7 三相机异分辨率(cam_high 848x480, cam_left/right 640x480), albumentations 链不能多视角同尺寸;
# m7_config 的 _get_vlm_inputs_uniform patch 会 center-crop 到 256x256, 故关 albumentations
export GR00T_USE_ALBUMENTATIONS=0
export https_proxy=http://192.168.16.68:18000
export http_proxy=http://192.168.16.68:18000

cd /home/chao01.wu/DeployAnything/GR00T_Dev/Isaac-GR00T

M7=/home/chao01.wu/DeployAnything/GR00T_Dev/M7_pickplace_example
OUT=/mnt/data/chao01.wu/M7_ft_cam_20m
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 --master_port=29501 \
  gr00t/experiment/launch_finetune.py \
  --base-model-path /home/chao01.wu/DeployAnything/GR00T_Dev/weights/GR00T-N1.7-3B \
  --dataset-path "$M7/2031605:$M7/2031607:$M7/2031704" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/M7/m7_config.py \
  --num-gpus 4 \
  --output-dir "$OUT" \
  --max-steps 312500 \
  --save-steps 50000 \
  --save-total-limit 8 \
  --global-batch-size 64 \
  --dataloader-num-workers 4 \
  --learning-rate 1e-4 \
  --warmup-ratio 0.05 \
  --weight-decay 1e-5 \
  2>&1 | tee "$OUT/train.log"
