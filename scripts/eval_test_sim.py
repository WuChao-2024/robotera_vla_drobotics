#!/usr/bin/env python3
import argparse
import io
import time
from typing import Any

import numpy as np
import torch
import zmq

from training.configs import config as _config
from training.interfaces.policies import policy_config as _policy_config
from training.interfaces.shared import download


STATE_DIM = 57
ACTION_DIM = 38
IMAGE_KEY_MAP = {
    "world_cam": "cam_high",
    "left_cam": "cam_left_wrist",
    "right_cam": "cam_right_wrist",
}


def _decode_npz_request(raw: bytes) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    with np.load(io.BytesIO(raw), allow_pickle=True) as npz:
        for key in npz.files:
            value = npz[key]
            if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
                value = value.item()
            payload[key] = value
    return payload


def _normalize_text_batch(text: Any, batch_size: int) -> list[str]:
    if text is None:
        return [""] * batch_size
    if isinstance(text, np.ndarray):
        text = text.tolist()
    if isinstance(text, str):
        return [text] * batch_size
    if isinstance(text, (list, tuple)):
        return [str(item) for item in text]
    return [str(text)] * batch_size


def _prepare_image_sample(image: np.ndarray, batch_index: int) -> np.ndarray:
    sample = image
    if sample.ndim == 4:
        sample = sample[batch_index]
    if sample.ndim != 3:
        raise ValueError(f"Expected image sample ndim=3, got shape {sample.shape}")

    if sample.shape[-1] == 3:
        sample = np.transpose(sample, (2, 0, 1))
    elif sample.shape[0] != 3:
        raise ValueError(f"Expected image with channel dim 3, got shape {sample.shape}")
    print(f"Prepared image sample with shape {sample.shape}")
    return np.asarray(sample, dtype=np.uint8)


def _prepare_images(image_dict: dict[str, np.ndarray], batch_index: int) -> dict[str, np.ndarray]:
    output = {}
    for key, value in image_dict.items():
        model_key = IMAGE_KEY_MAP.get(key, key)
        output[model_key] = _prepare_image_sample(np.asarray(value), batch_index)
    return output


class ModelInferenceServer:
    def __init__(self):
        self.policy = None
        self.load_model()

    def load_model(self):
        print(f"[{time.strftime('%H:%M:%S')}] loading model...")
        checkpoint_dir = download.maybe_download(
            "/era-ai/lm/user/wpc/openpi/checkpoints/pi05_M7_pp_opensource/260322/100000"
        )
        config = _config.get_config("pi05_M7_pp_opensource")
        self.policy = _policy_config.create_trained_policy(config, checkpoint_dir)
        print(f"[{time.strftime('%H:%M:%S')}] model ready")

    def infer_from_payload(self, payload: dict[str, Any]) -> np.ndarray:
        state = np.asarray(payload["state"], dtype=np.float32)
        if state.ndim == 1:
            state = state[None, :]
        if state.ndim != 2 or state.shape[1] != STATE_DIM:
            raise ValueError(f"Expected state shape [B,{STATE_DIM}], got {state.shape}")

        images = payload.get("images", {})
        if not isinstance(images, dict):
            raise ValueError("Payload field 'images' must be a dict.")
        text_batch = _normalize_text_batch(payload.get("text"), state.shape[0])
        if len(text_batch) < state.shape[0]:
            text_batch = text_batch + [""] * (state.shape[0] - len(text_batch))

        outputs = []
        for batch_index in range(state.shape[0]):
            model_input = {
                "state": state[batch_index],
                "images": _prepare_images(images, batch_index),
                "prompt": text_batch[batch_index],
            }
            with torch.no_grad():
                result = self.policy.infer(model_input)
            actions = result["actions"] if isinstance(result, dict) and "actions" in result else result
            actions = np.asarray(actions, dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
                raise ValueError(f"Expected model output shape [T,{ACTION_DIM}], got {actions.shape}")
            outputs.append(actions)
        return np.stack(outputs, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default="tcp://*:8003")
    args = parser.parse_args()

    server = ModelInferenceServer()
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(args.bind)
    print(f"[{time.strftime('%H:%M:%S')}] listening on {args.bind}")

    try:
        while True:
            raw = socket.recv()
            start = time.time()
            try:
                payload = _decode_npz_request(raw)
                actions = server.infer_from_payload(payload)
                socket.send_pyobj(actions.tolist())
                elapsed = time.time() - start
                print(
                    f"[{time.strftime('%H:%M:%S')}] served batch={actions.shape[0]} "
                    f"horizon={actions.shape[1]} in {elapsed:.3f}s"
                )
            except Exception as exc:
                socket.send_pyobj({"error": str(exc)})
    finally:
        socket.close(linger=0)
        context.term()


if __name__ == "__main__":
    main()
