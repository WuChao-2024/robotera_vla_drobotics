#!/usr/bin/env python3
"""
Pi05 HBM 模型推理服务 — ZMQ 服务端。

模型推理逻辑由 bpu_robotrea_pi05model.BPU_ROBOTREA_VLA_Pi05 提供，
本文件只负责 ZMQ 网络协议层。

协议与 pi05_bcModel_server.py 完全兼容，可用 compare.py 直接对比。

环境:
  conda activate robotrea_python_runtime

用法:
  python pi05_hbmModel_server.py --port 8006 --warmup
"""

import argparse
import io
import logging
import os
import time
import traceback

import numpy as np
import zmq

from bpu_robotrea_pi05model import BPU_ROBOTREA_VLA_Pi05, STATE_DIM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pi05_hbm_server")

# =============================================================================
# 默认路径
# =============================================================================

DEFAULT_MODEL_DIR = "/root/ssd/OELLM_Runtime/robotrea_model/M7_pickplace_example_ckpt_hbm_hbdk20260318"
DEFAULT_SIGLIP = os.path.join(DEFAULT_MODEL_DIR, "pi05_siglip_pi05_action_horizon_20_ptq.hbm")
DEFAULT_GEMMA_LLM = os.path.join(DEFAULT_MODEL_DIR, "pi05_gemma_llm_pi05_action_horizon_20_ptq.hbm")
DEFAULT_EXPERT = os.path.join(DEFAULT_MODEL_DIR, "pi05_gemma_expert_pi05_action_horizon_20_ptq.hbm")
DEFAULT_SHARE_DIR = "/root/ssd/OELLM_Runtime/robotrea_model/share"
DEFAULT_URDF_PATH = "/root/ssd/OELLM_Runtime/robotrea_model/share/l3_4.urdf"
DEFAULT_PROMPT = "pick the apple and put it in the bowl."

# =============================================================================
# 全局状态
# =============================================================================

_model = None


def load_model(siglip_path: str, gemma_llm_path: str, expert_path: str,
               share_dir: str, urdf_path: str = None):
    global _model
    _model = BPU_ROBOTREA_VLA_Pi05(
        siglip_path=siglip_path,
        gemma_llm_path=gemma_llm_path,
        expert_path=expert_path,
        share_dir=share_dir,
        urdf_path=urdf_path,
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
# ZMQ 服务（与 pi05_bcModel_server.py 协议一致）
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Pi05 HBM Model Inference Server (pyCauchyKesai)")
    p.add_argument("--siglip", type=str, default=DEFAULT_SIGLIP,
                   help="SigLIP .hbm model path")
    p.add_argument("--gemma-llm", type=str, default=DEFAULT_GEMMA_LLM,
                   help="Gemma LLM .hbm model path")
    p.add_argument("--expert", type=str, default=DEFAULT_EXPERT,
                   help="Gemma Expert .hbm model path")
    p.add_argument("--share-dir", type=str, default=DEFAULT_SHARE_DIR,
                   help="Shared resources dir (tokenizer, norm_stats, config.json)")
    p.add_argument("--urdf-path", type=str, default=DEFAULT_URDF_PATH,
                   help="URDF path for CamAbsoluteEeActions")
    p.add_argument("--port", type=int, default=45457)
    p.add_argument("--warmup", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    bind_addr = f"tcp://0.0.0.0:{args.port}"

    logger.info("=" * 60)
    logger.info("Pi05 HBM Model Inference Server (pyCauchyKesai)")
    logger.info(f"  SigLIP:     {args.siglip}")
    logger.info(f"  Gemma LLM:  {args.gemma_llm}")
    logger.info(f"  Expert:     {args.expert}")
    logger.info(f"  Share dir:  {args.share_dir}")
    logger.info(f"  URDF path:  {args.urdf_path or '(disabled)'}")
    logger.info(f"  Bind:       {bind_addr}")
    logger.info("=" * 60)

    load_model(args.siglip, args.gemma_llm, args.expert,
               args.share_dir, args.urdf_path)

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
