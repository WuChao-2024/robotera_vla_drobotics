"""M7 评测专用推理 server。

评测端 (star1-benchmark/scripts/vla_test_xbot.py) 用的是裸 ZMQ 协议:
    socket.send(np.savez_compressed(images=<dict>, state=<[N,57]>, text=<[str]>))
    reply = socket.recv_pyobj()           # 期望 {"actions": [B,T,38]} 或 [T,38]

而 GR00T 自带的 server (run_gr00t_server.py + PolicyClient) 是另一套 msgpack
协议 + get_action(observation) 端点, 两边线协议完全不兼容。本脚本在评测端
的裸协议上直接挂一个 Gr00tPolicy, 把两套协议桥接起来, 不动评测端任何代码。

桥接点 (全部依据 checkpoint-20000 里 new_embodiment 的 ground-truth 契约):
  1. images key 映射: 评测 cam_left_wrist/cam_right_wrist -> M7 cam_left/cam_right
  2. images 加时间维: [N,H,W,C] -> [N,1,H,W,C]  (video delta_indices=[0], 单帧)
  3. state[N,57] 按 new_embodiment 8 组切分 -> dict{key: [N,1,D]}
  4. text[str] -> {"annotation.human.task_description": [[t], ...]}  (B,1)
  5. get_action 返回 dict{right_end_pose,right_hand,left_end_pose,left_hand} 各
     [N,16,D] (decode_action 已 unnormalize 且 RELATIVE->绝对), 按 modality_keys
     顺序拼成 [N,16,38]
  6. socket.send_pyobj({"actions": arr})
"""
from __future__ import annotations

import argparse
import io
import os
import sys

import numpy as np
import torch
import zmq

# ---- 触发 m7_config 的两个 patch (Cosmos gated->本地副本, 多相机异分辨率 center-crop) ----
os.environ.setdefault(
    "GR00T_COSMOS_PATH",
    "/home/chao01.wu/DeployAnything/GR00T_Dev/weights/Cosmos-Reason2-2B",
)
REP = "/home/chao01.wu/DeployAnything/GR00T_Dev/Isaac-GR00T"
sys.path.insert(0, REP)
sys.path.insert(0, os.path.join(REP, "examples", "M7"))
import m7_config  # noqa: F401  触发 patch

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy

# 评测端 image key -> M7 new_embodiment video modality key
IMAGE_KEY_MAP = {
    "cam_high": "cam_high",
    "cam_left_wrist": "cam_left",
    "cam_right_wrist": "cam_right",
}
# new_embodiment state 切分 (statistics.json: right_arm_joint[7]+left_arm_joint[7]
# +right_end_pose[7]+left_end_pose[7]+right_hand[12]+left_hand[12]+waist[3]+neck[2]=57)
STATE_SLICES = [
    ("right_arm_joint", 0, 7),
    ("left_arm_joint", 7, 14),
    ("right_end_pose", 14, 21),
    ("left_end_pose", 21, 28),
    ("right_hand", 28, 40),
    ("left_hand", 40, 52),
    ("waist", 52, 55),
    ("neck", 55, 57),
]
# action modality_keys 顺序 (拼成 38 维, 与评测端 _decode_action_38_to_42 切分一致)
ACTION_KEYS = ["right_end_pose", "right_hand", "left_end_pose", "left_hand"]
LANGUAGE_KEY = "annotation.human.task_description"
STATE_DIM = 57


def build_observation(images_raw: dict, state_raw: np.ndarray, text_raw) -> dict:
    """评测端 payload -> Gr00tPolicy.get_action 期望的 observation 结构。"""
    # ---- video: {eval_key: [N,H,W,C] uint8} -> {m7_key: [N,1,H,W,C] uint8} ----
    # 推断 batch size (以 state 为准)
    state = np.asarray(state_raw, dtype=np.float32)
    if state.ndim == 1:
        state = state[None, :]
    N = state.shape[0]
    if state.shape[1] != STATE_DIM:
        raise ValueError(f"state dim expect {STATE_DIM}, got {state.shape[1]}")

    video = {}
    for eval_key, m7_key in IMAGE_KEY_MAP.items():
        if eval_key not in images_raw:
            raise ValueError(
                f"missing image key '{eval_key}'; got {list(images_raw.keys())}"
            )
        arr = np.asarray(images_raw[eval_key])
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        if arr.ndim == 3:  # [H,W,C] 单条 -> [1,H,W,C]
            arr = arr[None]
        if arr.ndim != 4:
            raise ValueError(f"image '{eval_key}' expect [N,H,W,C], got {arr.shape}")
        if arr.shape[0] != N:
            # 评测端 batch 由 env_ids 决定, 通常 N==1; 容错对齐
            arr = np.broadcast_to(arr, (N, *arr.shape[1:]))
        video[m7_key] = arr[:, None]  # [N,1,H,W,C]

    # ---- state: [N,57] -> dict{key: [N,1,D]} ----
    state_dict = {}
    for key, s, e in STATE_SLICES:
        state_dict[key] = state[:, s:e][:, None].astype(np.float32)  # [N,1,D]

    # ---- language: text=[str]xN -> {key: [[t], ...]} ----
    if isinstance(text_raw, np.ndarray):
        text_list = text_raw.tolist()
    elif isinstance(text_raw, str):
        text_list = [text_raw]
    else:
        text_list = list(text_raw)
    if len(text_list) != N:
        text_list = [text_list[0]] * N if text_list else [""] * N
    language = {LANGUAGE_KEY: [[str(t)] for t in text_list]}

    return {"video": video, "state": state_dict, "language": language}


def decode_reply(action_dict: dict) -> np.ndarray:
    """get_action 返回的 dict{key:[N,16,D]} -> [N,16,38] (按 ACTION_KEYS 顺序拼)。"""
    parts = []
    for key in ACTION_KEYS:
        if key not in action_dict:
            raise ValueError(f"action_dict missing key '{key}'; got {list(action_dict.keys())}")
        parts.append(np.asarray(action_dict[key], dtype=np.float32))
    actions = np.concatenate(parts, axis=-1)  # [N,16,38]
    return actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/mnt/data/chao01.wu/M7_ft/checkpoint-20000")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--strict", action="store_true", default=True)
    args = ap.parse_args()

    print(f"[serve_m7_eval] loading Gr00tPolicy from {args.model_path}", flush=True)
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        model_path=args.model_path,
        device=args.device,
        strict=args.strict,
    )
    print(f"[serve_m7_eval] policy ready on {args.device}", flush=True)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{args.host}:{args.port}")
    print(f"[serve_m7_eval] listening on tcp://{args.host}:{args.port}", flush=True)

    n_query = 0
    while True:
        msg = sock.recv()
        try:
            data = np.load(io.BytesIO(msg), allow_pickle=True)
            files = data.files
            # np.savez_compressed 把 dict/list 值存成 0-d object array
            images_raw = data["images"].item() if "images" in files else {}
            state_raw = data["state"]
            text_raw = data["text"] if "text" in files else ""
            if "seed" in files:
                torch.manual_seed(int(data["seed"]))  # 对拍模式：固定 init_noise（生产不带 seed 保持随机）

            observation = build_observation(images_raw, state_raw, text_raw)
            action_dict, _info = policy.get_action(observation)
            actions = decode_reply(action_dict)  # [N,16,38]

            n_query += 1
            a0 = actions[0, 0]  # 首条首帧
            print(
                f"[serve_m7_eval] q#{n_query} text={text_raw!r} "
                f"actions={actions.shape} "
                f"rpos={a0[0:3].round(3).tolist()} "
                f"rhand[min,max]=[{a0[7:19].min():.2f},{a0[7:19].max():.2f}]",
                flush=True,
            )
            sock.send_pyobj({"actions": actions})
        except Exception as e:
            import traceback
            traceback.print_exc()
            sock.send_pyobj({"error": f"serve_m7_eval: {e}"})


if __name__ == "__main__":
    main()
