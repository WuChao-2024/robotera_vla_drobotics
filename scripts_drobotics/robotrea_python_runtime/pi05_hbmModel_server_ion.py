#!/usr/bin/env python3
"""
Pi05 HBM 模型推理服务 — ION 零拷贝 + 可选 Butterworth 低通滤波。

使用 BPU_ROBOTREA_VLA_Pi05_ION 推理（最终版）：
  - KV Cache 零拷贝
  - SigLIP 3 实例多核并行
  - CPU/BPU 流水线重叠
  - 可选 Butterworth 低通滤波

用法:
  taskset -c 8-15 python pi05_hbmModel_server_ion.py \
    --urdf-path /root/ssd/OELLM_Runtime/robotrea_model/share/l3_4.urdf \
    --warmup --bind 10.112.10.106 --port 34345 \
    --filter-cutoff 1.0 --filter-fs 50.0 --filter-channels 16 --filter-order 2 --filter-steps 5
"""

import argparse
import io
import logging
import os
import time
import traceback

import numpy as np
import zmq

from bpu_robotrea_pi05model_ion import BPU_ROBOTREA_VLA_Pi05_ION, FrameEMA
from bpu_robotrea_pi05model import STATE_DIM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pi05_hbm_server_ion")

# =============================================================================
# 默认路径
# =============================================================================


DEFAULT_SIGLIP = "/root/ssd/OELLM_Runtime/robotrea_model/M7_pickplace_example_ckpt_hbm_v2_hbdk20260318/pi05_siglip_pi05_action_horizon_20_ptq.hbm"
DEFAULT_GEMMA_LLM = "/root/ssd/OELLM_Runtime/robotrea_model/M7_pickplace_example_ckpt_hbm_v2_hbdk20260318/pi05_gemma_llm_pi05_action_horizon_20_ptq.hbm"
DEFAULT_EXPERT = "/root/ssd/OELLM_Runtime/robotrea_model/M7_pickplace_example_ckpt_hbm_hbdk20260318/pi05_gemma_expert_pi05_action_horizon_20_ptq.hbm"
DEFAULT_SHARE_DIR = "/root/ssd/OELLM_Runtime/robotrea_model/share"
DEFAULT_URDF_PATH = "/root/ssd/OELLM_Runtime/robotrea_model/share/l3_4.urdf"
DEFAULT_PROMPT = "pick the apple and put it in the bowl."

# =============================================================================
# 全局状态
# =============================================================================

_model = None


def load_model(siglip_path: str, gemma_llm_path: str, expert_path: str,
               share_dir: str, urdf_path: str = None,
               filter_alpha: float = None, filter_channels: int = 38,
               filter_steps: int = 10):
    global _model
    _model = BPU_ROBOTREA_VLA_Pi05_ION(
        siglip_path=siglip_path,
        gemma_llm_path=gemma_llm_path,
        expert_path=expert_path,
        share_dir=share_dir,
        urdf_path=urdf_path,
        filter_alpha=filter_alpha,
        filter_channels=filter_channels,
        filter_steps=filter_steps,
    )


def infer(images, state, text):
    return _model.inference(images, state, text)


# =============================================================================
# Warmup
# =============================================================================

def warmup():
    logger.info("Warming up...")
    dummy_images = {
        "cam_high": np.random.randint(0, 256, (480, 848, 3), dtype=np.uint8),
        "cam_left_wrist": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
        "cam_right_wrist": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
    }
    dummy_state = np.random.randn(STATE_DIM).astype(np.float32)

    t0 = time.time()
    actions = infer(dummy_images, dummy_state, DEFAULT_PROMPT)
    elapsed = time.time() - t0
    logger.info(f"Warmup done in {elapsed:.1f}s. Output shape: {actions.shape}")


# =============================================================================
# ZMQ 服务
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Pi05 HBM Model Inference Server (ION + Filter)")
    p.add_argument("--siglip", type=str, default=DEFAULT_SIGLIP)
    p.add_argument("--gemma-llm", type=str, default=DEFAULT_GEMMA_LLM)
    p.add_argument("--expert", type=str, default=DEFAULT_EXPERT)
    p.add_argument("--share-dir", type=str, default=DEFAULT_SHARE_DIR)
    p.add_argument("--urdf-path", type=str, default=DEFAULT_URDF_PATH)
    p.add_argument("--bind", type=str, default="0.0.0.0",
                   help="Bind IP address (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=45457)
    p.add_argument("--warmup", action="store_true")
    # Filter 参数
    p.add_argument("--filter-alpha", type=float, default=None,
                   help="EMA smoothing factor (0-1), None=disabled. 0.5=moderate, 0.3=strong")
    p.add_argument("--filter-channels", type=int, default=38,
                   help="EMA channel count")
    p.add_argument("--filter-steps", type=int, default=10,
                   help="Smooth only first N steps (None=all)")
    return p.parse_args()


def main():
    args = parse_args()
    bind_addr = f"tcp://{args.bind}:{args.port}"

    logger.info("=" * 60)
    logger.info("Pi05 HBM Model Inference Server (ION + Filter)")
    logger.info(f"  SigLIP:     {args.siglip}")
    logger.info(f"  Gemma LLM:  {args.gemma_llm}")
    logger.info(f"  Expert:     {args.expert}")
    logger.info(f"  Share dir:  {args.share_dir}")
    logger.info(f"  URDF path:  {args.urdf_path or '(disabled)'}")
    logger.info(f"  Bind:       {bind_addr}")
    logger.info(f"  Filter:     alpha={args.filter_alpha or 'OFF'}, "
                f"ch={args.filter_channels}, "
                f"steps={args.filter_steps}")
    logger.info("=" * 60)

    load_model(args.siglip, args.gemma_llm, args.expert,
               args.share_dir, args.urdf_path,
               filter_alpha=args.filter_alpha,
               filter_channels=args.filter_channels,
               filter_steps=args.filter_steps)

    if args.warmup:
        warmup()

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(bind_addr)
    logger.info(f"Listening on {bind_addr}...")

    inference_count = 0
    total_time = 0.0

    try:
        while True:
            raw = socket.recv()
            buf = io.BytesIO(raw)
            data = np.load(buf, allow_pickle=True)

            images = data["images"].item() if (
                isinstance(data["images"], np.ndarray)
                and data["images"].dtype == object
            ) else data["images"]
            state = data["state"]
            text_list = data["text"]

            if state.ndim == 1:
                B = 1
                states_batch = state[np.newaxis, :]
                texts_batch = [text_list] if isinstance(text_list, str) else text_list
                images_batch = {
                    k: (v[np.newaxis, ...] if v.ndim == 3 else v)
                    for k, v in images.items()
                }
            else:
                B = state.shape[0]
                states_batch = state
                texts_batch = text_list if isinstance(text_list, list) else [text_list]
                images_batch = images

            all_actions = []
            t_start = time.time()

            for i in range(B):
                img_dict = {k: v[i] if v.ndim == 4 else v
                           for k, v in images_batch.items()}
                s = states_batch[i]
                t = texts_batch[i] if isinstance(texts_batch, list) else texts_batch
                if not t or (isinstance(t, float) and np.isnan(t)):
                    t = DEFAULT_PROMPT

                actions = infer(img_dict, s, str(t))
                all_actions.append(actions)

            elapsed = time.time() - t_start
            total_time += elapsed
            inference_count += 1

            actions_array = np.stack(all_actions, axis=0).astype(np.float32)
            socket.send_pyobj({"actions": actions_array})

            if inference_count % 10 == 0:
                avg = total_time / inference_count
                logger.info(
                    f"[{inference_count}] avg={avg:.3f}s, last={elapsed:.3f}s, "
                    f"shape={actions_array.shape}"
                )

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
    finally:
        socket.close()
        context.term()
        logger.info(f"Server stopped. Total inferences: {inference_count}")


if __name__ == "__main__":
    main()