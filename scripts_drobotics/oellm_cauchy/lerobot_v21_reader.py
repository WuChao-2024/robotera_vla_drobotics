#!/usr/bin/env python3
"""
LeRobot Dataset v2.1 校准数据读取器。

数据管线与 compare.py 完全一致：
  - meta/tasks.jsonl 读取 prompt
  - data/chunk-000/*.parquet 读取 state / action / task_index
  - videos/chunk-000/{cam_key}/{ep_name}.mp4 + cv2.VideoCapture 读取帧
  - BGR → RGB: frame[..., ::-1]

优化：先读 parquet 元数据（快），打乱采样后只解码需要的帧，
避免全量 286455 帧视频解码。
"""

import json
import os
import random

import cv2
import numpy as np
import pyarrow.parquet as pq

# 相机 key（与 compare.py 一致）
CAM_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left",
    "observation.images.cam_right",
]
OUTPUT_KEYS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


def load_tasks(dataset_root: str) -> dict:
    """从 meta/tasks.jsonl 读取任务文字（与 compare.py 一致）。"""
    tasks_path = os.path.join(dataset_root, "meta", "tasks.jsonl")
    tasks = {}
    with open(tasks_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            tasks[item["task_index"]] = item["task"]
    return tasks


def collect_episodes(dataset_root: str) -> list[str]:
    """收集 episode parquet 路径（与 compare.py 一致）。"""
    data_dir = os.path.join(dataset_root, "data", "chunk-000")
    return sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".parquet")
    )


def _scan_frame_metadata(dataset_root: str) -> list[dict]:
    """只读 parquet 元数据，不解码视频（快）。

    Returns:
        List of metadata dicts:
            {
                "ep_name": str,
                "frame_index": int,
                "task_index": int,
                "episode_index": int,
                "state": np.ndarray float32,
                "action": np.ndarray float32,
            }
    """
    episodes = collect_episodes(dataset_root)
    print(f"  Found {len(episodes)} episodes, scanning metadata...")

    meta_list = []
    for ep_path in episodes:
        ep_name = os.path.splitext(os.path.basename(ep_path))[0]

        # 检查 3 路视频是否都存在（与 compare.py 一致）
        video_dir = os.path.join(dataset_root, "videos", "chunk-000")
        all_exist = True
        for cam_key in CAM_KEYS:
            vp = os.path.join(video_dir, cam_key, f"{ep_name}.mp4")
            if not os.path.exists(vp):
                all_exist = False
                break
        if not all_exist:
            continue

        table = pq.read_table(ep_path)
        df = table.to_pandas()

        for row_idx in range(len(df)):
            row = df.iloc[row_idx]
            meta_list.append({
                "ep_name": ep_name,
                "frame_index": int(row["frame_index"]),
                "task_index": int(row["task_index"]),
                "episode_index": int(row["episode_index"]),
                "state": row["observation.state"].astype(np.float32),
                "action": row["action"].astype(np.float32),
            })

    return meta_list


def _decode_sample(
    dataset_root: str,
    meta: dict,
    tasks: dict,
) -> dict | None:
    """根据元数据解码单帧视频（与 compare.py 一致：cv2 + BGR→RGB）。"""
    video_dir = os.path.join(dataset_root, "videos", "chunk-000")
    ep_name = meta["ep_name"]
    frame_idx = meta["frame_index"]

    images = {}
    for cam_key, out_key in zip(CAM_KEYS, OUTPUT_KEYS):
        video_path = os.path.join(video_dir, cam_key, f"{ep_name}.mp4")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        cap.set(1, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        images[out_key] = frame[..., ::-1]  # BGR → RGB

    return {
        "images": images,
        "state": meta["state"],
        "prompt": tasks.get(meta["task_index"], ""),
        "action": meta["action"],
        "task_index": meta["task_index"],
        "episode_index": meta["episode_index"],
        "frame_index": frame_idx,
    }


def load_calib_samples(
    dataset_root: str,
    cal_num: int,
    seed: int = 42,
) -> list[dict]:
    """从 LeRobot v2.1 数据集加载校准样本。

    数据管线与 compare.py 完全一致，但只解码采样到的帧：
      1. 读 parquet 元数据（快，无视频解码）
      2. 打乱 + 取前 cal_num 条
      3. 只解码这些帧的视频

    Args:
        dataset_root: v2.1 数据集根目录
        cal_num: 校准样本数
        seed: 随机种子

    Returns:
        List of sample dicts（格式与 compare.py 一致）
    """
    print(f"  Loading dataset from {dataset_root} ...")
    tasks = load_tasks(dataset_root)
    meta_list = _scan_frame_metadata(dataset_root)
    print(f"  Total: {len(meta_list)} frames (metadata only)")

    # 打乱 + 采样
    rng = random.Random(seed)
    rng.shuffle(meta_list)

    actual_num = min(cal_num, len(meta_list))
    if actual_num < cal_num:
        print(f"  ⚠ Requested {cal_num} samples, but only {len(meta_list)} available")

    selected_meta = meta_list[:actual_num]
    print(f"  Decoding {actual_num} calibration frames (seed={seed})...")

    samples = []
    for i, meta in enumerate(selected_meta):
        sample = _decode_sample(dataset_root, meta, tasks)
        if sample is not None:
            samples.append(sample)
        if (i + 1) % 10 == 0:
            print(f"    Decoded {i + 1}/{actual_num}")

    print(f"  Got {len(samples)} valid calibration samples")
    return samples
