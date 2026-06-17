#!/usr/bin/env python3
"""
Pi05 LeapModel 推理服务 — 用于验证 leap 模型与 torch 模型的一致性。

加载 pi05 的 safetensors 权重到 leap 模型结构中，
使用 compile_mode(False) 的 PyTorch forward 路径做推理，
协议与 pi05_torch_server.py 完全兼容，可用 compare.py 直接对比。

完整的 M7 pi05 数据路径（对齐 torch 推理 pipeline）：
  输入: HWC uint8 images + raw state + prompt
    → M7Inputs (pad→640x480, rename cameras)
    → Normalize (state z-score)
    → Tokenize + Siglip → Gemma LLM prefix → Gemma Expert 10-step denoising
  输出: raw model actions
    → Unnormalize (action z-score inverse)
    → CamAbsoluteEeActions (delta camera→absolute base, URDF kinematics)
    → M7Outputs (extract action_dim)

用法:
  conda activate robotera_vla_drobotics_convert
  cd /path/to/oellm_cauchy
  python pi05_leapModel_server.py \
      --model-path /path/to/safetensors_model \
      --port 8005 --warmup
"""

import argparse
import io
import logging
import os
import sys
import time
import traceback

import numpy as np
import zmq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pi05_leap")

# =============================================================================
# 路径与常量
# =============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

DEFAULT_MODEL_PATH = "/home/chao.wu/pi_oellm/ROBOTREA/M7_pickplace_example_ckpt_pt"
DEFAULT_PROMPT = "pick the apple and put it in the bowl."
DEFAULT_URDF_PATH = "/home/chao.wu/pi_oellm/ROBOTREA/robotera_vla_drobotics/l3_4.urdf"

# =============================================================================
# 全局模型状态
# =============================================================================

_model = None


# =============================================================================
# 模型加载
# =============================================================================

def load_model(model_path, device="cuda"):
    """加载模型。"""
    global _model

    from pi05_leap_model import Pi05LeapModel

    safetensors_file = os.path.join(model_path, "model.safetensors")
    if not os.path.isfile(safetensors_file):
        raise FileNotFoundError(f"model.safetensors not found in {model_path}")

    logger.info(f"Loading Pi05LeapModel from {model_path} (device={device})...")
    _model = Pi05LeapModel(model_path, urdf_path=DEFAULT_URDF_PATH,
                            action_horizon=20, device=device)
    logger.info(f"Model loaded. action_dim={_model.action_dim}")


# =============================================================================
# 推理
# =============================================================================

def infer(images, state, text):
    """单次推理。完整对齐 torch M7 pi05 推理 pipeline。"""
    return _model.infer(images, state, text)


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
    dummy_state = np.random.randn(57).astype(np.float32)

    t0 = time.time()
    actions = infer(dummy_images, dummy_state, DEFAULT_PROMPT)
    elapsed = time.time() - t0
    logger.info(f"Warmup done in {elapsed:.1f}s. Output shape: {actions.shape}")


# =============================================================================
# ZMQ 服务（与 pi05_torch_server.py 协议一致）
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Pi05 LeapModel Inference Server")
    p.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--port", type=int, default=45456)
    p.add_argument("--warmup", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    bind_addr = f"tcp://0.0.0.0:{args.port}"

    logger.info("=" * 60)
    logger.info("Pi05 LeapModel Inference Server")
    logger.info(f"  Model:    {args.model_path}")
    logger.info(f"  Device:   {args.device}")
    logger.info(f"  Bind:     {bind_addr}")
    logger.info("=" * 60)

    load_model(args.model_path, args.device)

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
                img_dict = {k: v[i] if v.ndim == 4 else v for k, v in images_batch.items()}
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
