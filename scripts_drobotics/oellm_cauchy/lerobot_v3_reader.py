#!/usr/bin/env python3
"""
LeRobot Dataset v3.0 轻量读取器。

不依赖 lerobot 包本身（需要 Python>=3.12），使用 pyarrow + cv2
实现 v3.0 格式数据集的读取，读取方式与 compare.py 完全对齐：
  - cv2.VideoCapture + cap.set(1, frame_index) seek 读取视频帧
  - BGR → RGB: frame[..., ::-1]
  - parquet 读取 observation.state / action / task_index
  - 自动发现子数据集目录
  - 全局打乱 + 采样
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq


# 相机 key 列表（与 compare.py 一致）
CAM_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left",
    "observation.images.cam_right",
]


# =============================================================================
# 数据集发现
# =============================================================================

def discover_lerobot_v3_datasets(dataset_root: str) -> list[str]:
    """发现 LeRobot v3.0 数据集。

    支持两种目录结构：
    1. dataset_root 本身就是一个数据集（含 meta/info.json）
    2. dataset_root 下有多个子数据集目录

    Returns:
        List of dataset root paths
    """
    root = Path(dataset_root)

    if (root / "meta" / "info.json").exists():
        return [str(root)]

    subdirs = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "meta" / "info.json").exists():
            subdirs.append(str(d))

    if not subdirs:
        raise FileNotFoundError(
            f"在 {dataset_root} 下未找到 LeRobot v3.0 数据集 "
            f"（需要 meta/info.json）"
        )

    return subdirs


# =============================================================================
# 单个 v3.0 数据集读取
# =============================================================================

class LeRobotV3Dataset:
    """轻量的 LeRobot v3.0 数据集读取器。

    使用 cv2.VideoCapture 读取视频帧（与 compare.py 一致），
    通过 episode metadata 定位视频文件。
    """

    def __init__(self, root: str):
        self.root = Path(root)
        self._load_metadata()
        self._load_data()
        self._load_episode_meta()

    def _load_metadata(self):
        info_path = self.root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"meta/info.json not found in {self.root}")
        with open(info_path) as f:
            self.info = json.load(f)

        self.fps = self.info["fps"]
        self.total_frames = self.info["total_frames"]
        self.total_episodes = self.info["total_episodes"]

        # 加载 tasks（v3.0 用 tasks.parquet）
        tasks_path = self.root / "meta" / "tasks.parquet"
        self.tasks = {}
        if tasks_path.exists():
            tasks_df = pq.read_table(str(tasks_path)).to_pandas()
            # v3.0: task 文字是 index，task_index 是列
            if "task" not in tasks_df.columns and tasks_df.index.name == "task":
                tasks_df = tasks_df.reset_index()
            for _, row in tasks_df.iterrows():
                task_idx = int(row["task_index"])
                task_text = str(row["task"])
                self.tasks[task_idx] = task_text

    def _load_data(self):
        """加载所有 parquet 数据文件。"""
        import pandas as pd

        data_dir = self.root / "data"
        all_dfs = []
        for chunk_dir in sorted(data_dir.iterdir()):
            if not chunk_dir.is_dir():
                continue
            for pq_file in sorted(chunk_dir.glob("*.parquet")):
                table = pq.read_table(str(pq_file))
                all_dfs.append(table.to_pandas())

        self._df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        self._len = len(self._df)

    def _load_episode_meta(self):
        """加载 episode 元数据（用于视频帧定位）。"""
        import pandas as pd

        ep_dir = self.root / "meta" / "episodes"
        all_ep_dfs = []
        if ep_dir.exists():
            for chunk_dir in sorted(ep_dir.iterdir()):
                if not chunk_dir.is_dir():
                    continue
                for pq_file in sorted(chunk_dir.glob("*.parquet")):
                    table = pq.read_table(str(pq_file))
                    all_ep_dfs.append(table.to_pandas())

        self._ep_df = pd.concat(all_ep_dfs, ignore_index=True) if all_ep_dfs else None

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int) -> dict:
        """读取第 idx 帧，返回与 compare.py 一致格式的 sample dict。

        Returns:
            {
                "images": {
                    "cam_high":        np.ndarray [H, W, 3] RGB uint8,
                    "cam_left_wrist":  np.ndarray [H, W, 3] RGB uint8,
                    "cam_right_wrist": np.ndarray [H, W, 3] RGB uint8,
                },
                "state":   np.ndarray float32,
                "prompt":  str,
                "action":  np.ndarray float32,
                "task_index":     int,
                "episode_index":  int,
                "frame_index":    int,
            }
        """
        if idx < 0 or idx >= self._len:
            raise IndexError(f"Index {idx} out of range [0, {self._len})")

        row = self._df.iloc[idx]
        episode_index = int(row["episode_index"])
        frame_index = int(row["frame_index"])
        task_index = int(row["task_index"])

        # --- 使用 cv2.VideoCapture 读取视频帧（与 compare.py 一致）---
        images = {}
        if self._ep_df is not None:
            ep_row = self._ep_df[self._ep_df["episode_index"] == episode_index]
            if len(ep_row) == 0:
                raise ValueError(f"Episode {episode_index} not found in episode metadata")
            ep_row = ep_row.iloc[0]

            # 输出 key 映射（与 compare.py 一致）
            output_keys = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

            for cam_key, out_key in zip(CAM_KEYS, output_keys):
                vid_chunk = int(ep_row[f"videos/{cam_key}/chunk_index"])
                vid_file = int(ep_row[f"videos/{cam_key}/file_index"])
                video_path = str(self.root / "videos" / cam_key /
                                 f"chunk-{vid_chunk:03d}" / f"file-{vid_file:03d}.mp4")

                cap = cv2.VideoCapture(video_path)
                cap.set(1, frame_index)
                ret, frame = cap.read()
                cap.release()

                if not ret:
                    raise RuntimeError(
                        f"Failed to read frame {frame_index} from {video_path}"
                    )
                images[out_key] = frame[..., ::-1]  # BGR → RGB

        # --- 读取数值数据（与 compare.py 一致）---
        state = row["observation.state"].astype(np.float32)
        action = row["action"].astype(np.float32)
        prompt = self.tasks.get(task_index, "")

        return {
            "images": images,
            "state": state,
            "prompt": prompt,
            "action": action,
            "task_index": task_index,
            "episode_index": episode_index,
            "frame_index": frame_index,
        }


# =============================================================================
# 校准数据采样
# =============================================================================

def load_calib_samples(
    dataset_root: str,
    cal_num: int,
    seed: int = 42,
) -> list[dict]:
    """从 LeRobot v3.0 数据集加载校准样本。

    自动发现子数据集，合并所有帧，打乱后取前 cal_num 条。

    Args:
        dataset_root: v3.0 数据集根目录
        cal_num: 校准样本数
        seed: 随机种子

    Returns:
        List of sample dicts (与 LeRobotV3Dataset.__getitem__ 返回格式一致)
    """
    import random

    dataset_paths = discover_lerobot_v3_datasets(dataset_root)
    print(f"  发现 {len(dataset_paths)} 个子数据集:")

    datasets = []
    total_frames = 0
    for path in dataset_paths:
        ds_name = os.path.basename(path)
        ds = LeRobotV3Dataset(path)
        datasets.append(ds)
        total_frames += len(ds)
        print(f"    '{ds_name}': {len(ds)} 帧")

    print(f"  合计: {total_frames} 帧")

    # 收集所有 (ds_idx, frame_idx) 对
    all_indices = []
    for ds_idx, ds in enumerate(datasets):
        for frame_idx in range(len(ds)):
            all_indices.append((ds_idx, frame_idx))

    # 打乱
    rng = random.Random(seed)
    rng.shuffle(all_indices)

    # 采样
    actual_num = min(cal_num, len(all_indices))
    if actual_num < cal_num:
        print(f"  ⚠ 请求 {cal_num} 条，但只有 {len(all_indices)} 条可用")

    selected = all_indices[:actual_num]
    print(f"  加载 {actual_num} 条校准样本...")

    samples = []
    for ds_idx, frame_idx in selected:
        sample = datasets[ds_idx][frame_idx]
        samples.append(sample)

    return samples
