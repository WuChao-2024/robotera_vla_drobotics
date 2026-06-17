#!/usr/bin/env python3
"""
Pi05 PyTorch 推理服务（协议兼容 star1_vla_inference）。

协议：
  请求: NPZ 压缩字节
    - images:  {cam_high, cam_left_wrist, cam_right_wrist} HWC uint8
    - state:   (57,) float32
    - text:    str
  响应: send_pyobj({"actions": ndarray (B, 20, 38) float32})

用法:
  python pi05_torch_server.py --port 8004 --warmup
"""
import argparse
import io
import logging
import os
import sys
import time
import traceback

import numpy as np
import torch
import zmq

# 禁用 JAX GPU 预分配。torch 服务会导入 training 模块，间接触发 JAX 初始化。
# JAX 默认预分配 75% 显存，与 PyTorch 模型争抢导致 OOM。
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pi05_torch")

# =============================================================================
# 路径常量
# =============================================================================

ROBOTREA_ROOT = "/home/chao.wu/pi_oellm/ROBOTREA/robotera_vla_drobotics"
CHECKPOINT_DIR = "/home/chao.wu/pi_oellm/ROBOTREA/M7_pickplace_example_ckpt_pt"
POLICY_NAME = "pi05_M7_pp_opensource"
DEFAULT_PROMPT = "pick the apple and put it in the bowl."

_HWC_TO_CHW = (2, 0, 1)

# =============================================================================
# 模型加载
# =============================================================================

_policy = None
_config_module = None
_policy_config_module = None
_download_module = None


def _ensure_imports():
    global _config_module, _policy_config_module, _download_module

    if _config_module is not None:
        return

    if ROBOTREA_ROOT not in sys.path:
        sys.path.insert(0, ROBOTREA_ROOT)

    _original_cwd = os.getcwd()
    os.chdir(ROBOTREA_ROOT)
    try:
        from training.configs import config as _c
        from training.interfaces.policies import policy_config as _pc
        from training.interfaces.shared import download as _dl
    finally:
        os.chdir(_original_cwd)

    _config_module = _c
    _policy_config_module = _pc
    _download_module = _dl


def load_model():
    global _policy

    _ensure_imports()

    checkpoint_dir = _download_module.maybe_download(CHECKPOINT_DIR)
    train_config = _config_module.get_config(POLICY_NAME)

    logger.info(f"Loading policy '{POLICY_NAME}' from {checkpoint_dir}")
    _policy = _policy_config_module.create_trained_policy(
        train_config, checkpoint_dir,
        sample_kwargs={"num_steps": 10},
        default_prompt=DEFAULT_PROMPT,
        pytorch_device="cuda",
    )
    logger.info("Policy loaded.")


def infer(images, state, text):
    """单次 PyTorch 推理."""
    if _policy is None:
        raise RuntimeError("Model not loaded.")

    imgs_chw = {
        "cam_high": np.transpose(images["cam_high"], _HWC_TO_CHW),
        "cam_left_wrist": np.transpose(images["cam_left_wrist"], _HWC_TO_CHW),
        "cam_right_wrist": np.transpose(images["cam_right_wrist"], _HWC_TO_CHW),
    }

    inputs = {
        "images": imgs_chw,
        "state": state.astype(np.float32),
        "prompt": text,
    }

    torch.manual_seed(42)
    result = _policy.infer(inputs)
    return result["actions"]


def warmup():
    logger.info("Warming up...")
    dummy_images = {
        "cam_high": np.random.randint(0, 256, (480, 848, 3), dtype=np.uint8),
        "cam_left_wrist": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
        "cam_right_wrist": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
    }
    dummy_state = np.random.randn(57).astype(np.float32)
    dummy_text = DEFAULT_PROMPT

    t0 = time.time()
    actions = infer(dummy_images, dummy_state, dummy_text)
    elapsed = time.time() - t0
    logger.info(f"Warmup done in {elapsed:.1f}s. Output shape: {actions.shape}")


# =============================================================================
# ZMQ 服务
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Pi05 PyTorch Inference Server")
    p.add_argument("--port", type=int, default=45455)
    p.add_argument("--warmup", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    bind_addr = f"tcp://0.0.0.0:{args.port}"

    logger.info("=" * 60)
    logger.info(f"Pi05 PyTorch Inference Server")
    logger.info(f"  Checkpoint:  {CHECKPOINT_DIR}")
    logger.info(f"  Policy:      {POLICY_NAME}")
    logger.info(f"  Bind addr:   {bind_addr}")
    logger.info("=" * 60)

    load_model()

    if args.warmup:
        warmup()

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(bind_addr)
    logger.info(f"Listening on {bind_addr}...")

    inference_count = 0
    total_inference_time = 0.0

    try:
        while True:
            raw = socket.recv()
            buf = io.BytesIO(raw)
            data = np.load(buf, allow_pickle=True)

            images = data["images"].item() if isinstance(data["images"], np.ndarray) and data["images"].dtype == object else data["images"]
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
            total_inference_time += elapsed
            inference_count += 1

            actions_array = np.stack(all_actions, axis=0).astype(np.float32)
            socket.send_pyobj({"actions": actions_array})

            if inference_count % 10 == 0:
                avg_time = total_inference_time / inference_count
                logger.info(f"[{inference_count}] avg={avg_time:.3f}s, last={elapsed:.3f}s, shape={actions_array.shape}")

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
