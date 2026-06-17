#!/usr/bin/env python3
"""
Pi05 JAX vs PyTorch 推理对比工具。

以客户端方式分别请求 JAX 和 PyTorch 推理服务，对比输出 action chunk。

用法:
  # 先启动两个服务（协议兼容 star1_vla_inference）
  python pi05_jax_server.py --port 8003 --warmup &
  python pi05_torch_server.py --port 8004 --warmup &

  # 运行对比
  python compare.py
  python compare.py --frame-count 500
"""

import argparse
import csv
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import zmq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pi05_compare")

# =============================================================================
# 默认参数
# =============================================================================

DEFAULT_DATASET_ROOT = "/home/chao.wu/pi_oellm/ROBOTREA/M7_pickplace_example_merged_LeRobotDataSetv2.1"
DEFAULT_JAX_SERVER = "tcp://localhost:8003"
DEFAULT_TORCH_SERVER = "tcp://localhost:8004"
DEFAULT_FRAME_COUNT = 100
DEFAULT_OUTPUT_DIR = "./comparison_results"

ACTION_HORIZON = 20
ACTION_DIM = 38

_HWC_TO_CHW = (2, 0, 1)


# =============================================================================
# 数据集加载（LeRobot v2.1）
# =============================================================================

def load_tasks(dataset_root):
    tasks_path = os.path.join(dataset_root, "meta", "tasks.jsonl")
    tasks = {}
    with open(tasks_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            tasks[item["task_index"]] = item["task"]
    return tasks


def collect_episodes(dataset_root):
    data_dir = os.path.join(dataset_root, "data", "chunk-000")
    episodes = sorted(
        [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".parquet")]
    )
    return episodes


def load_frames(dataset_root, frame_count, tasks):
    """从数据集顺序读取前 frame_count 帧."""
    import pyarrow.parquet as pq
    import cv2

    episodes = collect_episodes(dataset_root)
    logger.info(f"Found {len(episodes)} episodes")

    video_dir = os.path.join(dataset_root, "videos", "chunk-000")
    cam_keys = ["observation.images.cam_high", "observation.images.cam_left", "observation.images.cam_right"]
    output_keys = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

    frames = []
    for ep_path in episodes:
        if len(frames) >= frame_count:
            break

        ep_name = os.path.splitext(os.path.basename(ep_path))[0]
        table = pq.read_table(ep_path)
        df = table.to_pandas()

        caps = {}
        for cam_key, out_key in zip(cam_keys, output_keys):
            video_path = os.path.join(video_dir, cam_key, f"{ep_name}.mp4")
            if not os.path.exists(video_path):
                continue
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                caps[out_key] = cap

        if len(caps) != 3:
            for c in caps.values():
                c.release()
            continue

        for row_idx in range(len(df)):
            if len(frames) >= frame_count:
                break

            row = df.iloc[row_idx]
            frame_idx_in_episode = int(row["frame_index"])

            images = {}
            decode_ok = True
            for out_key, cap in caps.items():
                cap.set(1, frame_idx_in_episode)
                ret, frame = cap.read()
                if not ret:
                    decode_ok = False
                    break
                images[out_key] = frame[..., ::-1]  # BGR → RGB

            if not decode_ok:
                continue

            task_index = int(row["task_index"])
            prompt = tasks.get(task_index, "")

            frames.append({
                "images": images,
                "state": row["observation.state"].astype(np.float32),
                "prompt": prompt,
                "ground_truth_action": row["action"].astype(np.float32),
                "meta": {
                    "episode_index": int(row["episode_index"]),
                    "frame_index": int(row["frame_index"]),
                    "task_index": task_index,
                    "episode_name": ep_name,
                },
            })

        for c in caps.values():
            c.release()

    logger.info(f"Loaded {len(frames)} frames")
    return frames


# =============================================================================
# ZMQ 客户端（每次重新连接，避免 REQ 状态损坏）
# =============================================================================

class InferenceClient:
    """ZMQ 推理客户端。每次请求创建新连接，避免超时导致的状态损坏。"""

    def __init__(self, server_addr: str, timeout: int = 300000):
        self.server_addr = server_addr
        self.timeout = timeout

    def infer(self, images: dict, state: np.ndarray, text: str) -> np.ndarray:
        """发送推理请求，返回 actions (20, 38). 每次创建新 socket."""
        ctx = zmq.Context()
        socket = ctx.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.server_addr)

        try:
            # 发送 HWC，与 star1_vla_inference 协议一致
            npz_data = {
                "images": images,
                "state": state.astype(np.float32),
                "text": text,
            }

            buf = io.BytesIO()
            np.savez_compressed(buf, **npz_data)
            socket.send(buf.getvalue())
            reply = socket.recv_pyobj()
            return reply["actions"]
        finally:
            socket.close()
            ctx.term()


# =============================================================================
# 指标计算
# =============================================================================

def compute_metrics(a_jax, a_torch, ground_truth=None):
    """对比两个 action 数组."""
    diff = a_jax - a_torch
    abs_diff = np.abs(diff)

    mse = float(np.mean(diff ** 2))
    cos_sims = []
    for t in range(a_jax.shape[0]):
        dot = np.dot(a_jax[t], a_torch[t])
        norm = np.linalg.norm(a_jax[t]) * np.linalg.norm(a_torch[t]) + 1e-8
        cos_sims.append(float(dot / norm))

    metrics = {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(abs_diff)),
        "max_abs_error": float(np.max(abs_diff)),
        "rel_error": float(np.mean(abs_diff / np.maximum(np.abs(a_jax), 1e-8))),
        "cosine_similarity_mean": float(np.mean(cos_sims)),
        "agreement_1e-2": float(np.mean(abs_diff < 1e-2)),
        "agreement_1e-3": float(np.mean(abs_diff < 1e-3)),
        "agreement_1e-4": float(np.mean(abs_diff < 1e-4)),
        "agreement_1e-5": float(np.mean(abs_diff < 1e-5)),
    }

    if ground_truth is not None:
        if ground_truth.ndim == 1:
            gt = ground_truth[np.newaxis, :]
        else:
            gt = ground_truth
        metrics["mse_jax_gt"] = float(np.mean((a_jax - gt) ** 2))
        metrics["mse_torch_gt"] = float(np.mean((a_torch - gt) ** 2))

    return metrics


# =============================================================================
# CLI & Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="JAX vs PyTorch 推理对比工具")
    p.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--jax-server", type=str, default=DEFAULT_JAX_SERVER)
    p.add_argument("--torch-server", type=str, default=DEFAULT_TORCH_SERVER)
    p.add_argument("--frame-count", type=int, default=DEFAULT_FRAME_COUNT)
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Pi05 JAX vs PyTorch 推理对比")
    logger.info(f"  Dataset:   {args.dataset_root}")
    logger.info(f"  JAX:       {args.jax_server}")
    logger.info(f"  Torch:     {args.torch_server}")
    logger.info(f"  Frames:    {args.frame_count}")
    logger.info(f"  Output:    {output_dir}")
    logger.info("=" * 60)

    # 1. 加载数据集
    logger.info("Loading dataset...")
    t0 = time.time()
    tasks = load_tasks(args.dataset_root)
    frames = load_frames(args.dataset_root, args.frame_count, tasks)
    logger.info(f"Loaded {len(frames)} frames in {time.time() - t0:.1f}s")
    if len(frames) == 0:
        logger.error("No frames loaded.")
        sys.exit(1)

    # 2. 创建两个客户端
    client_jax = InferenceClient(args.jax_server)
    client_torch = InferenceClient(args.torch_server)

    # 3. 逐帧对比
    all_metrics = []
    total_time_jax = 0.0
    total_time_torch = 0.0
    success_count = 0

    for i, frame in enumerate(frames):
        images = frame["images"]
        state = frame["state"]
        prompt = frame["prompt"]
        gt_action = frame["ground_truth_action"]

        # 请求 JAX
        t1 = time.time()
        try:
            actions_jax = np.squeeze(client_jax.infer(images, state, prompt))
            time_jax = time.time() - t1
        except Exception as e:
            logger.error(f"[{i}] JAX failed: {e}")
            continue

        # 请求 PyTorch
        t2 = time.time()
        try:
            actions_torch = np.squeeze(client_torch.infer(images, state, prompt))
            time_torch = time.time() - t2
        except Exception as e:
            logger.error(f"[{i}] Torch failed: {e}")
            continue

        total_time_jax += time_jax
        total_time_torch += time_torch
        success_count += 1

        metrics = compute_metrics(actions_jax, actions_torch, ground_truth=gt_action)
        metrics["sample_index"] = i
        metrics["episode_name"] = frame["meta"]["episode_name"]
        metrics["frame_index"] = int(frame["meta"]["frame_index"])
        metrics["task_index"] = int(frame["meta"]["task_index"])
        metrics["prompt"] = prompt
        metrics["time_jax_ms"] = time_jax * 1000
        metrics["time_torch_ms"] = time_torch * 1000
        all_metrics.append(metrics)

        if (i + 1) % 10 == 0:
            logger.info(
                f"[{i + 1}/{len(frames)}] "
                f"MSE={metrics['mse']:.2e}, "
                f"CosSim={metrics['cosine_similarity_mean']:.6f}, "
                f"JAX={time_jax * 1000:.0f}ms, Torch={time_torch * 1000:.0f}ms"
            )

    # 4. 输出
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Done. {success_count}/{len(frames)} samples compared.")

    if success_count == 0:
        logger.error("All samples failed. No output generated.")
        sys.exit(1)

    logger.info(f"Total JAX time:   {total_time_jax:.2f}s ({total_time_jax / success_count * 1000:.0f}ms avg)")
    logger.info(f"Total Torch time: {total_time_torch:.2f}s ({total_time_torch / success_count * 1000:.0f}ms avg)")

    metric_keys = ["mse", "rmse", "mae", "max_abs_error", "rel_error", "cosine_similarity_mean",
                   "agreement_1e-2", "agreement_1e-3", "agreement_1e-4", "agreement_1e-5"]
    extra_keys = ["mse_jax_gt", "mse_torch_gt", "time_jax_ms", "time_torch_ms"]
    all_keys = ["sample_index", "episode_name", "frame_index", "task_index", "prompt"] + metric_keys + extra_keys
    existing_keys = [k for k in all_keys if k in all_metrics[0]]

    # per_sample.csv
    csv_path = output_dir / "per_sample.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=existing_keys)
        writer.writeheader()
        writer.writerows(all_metrics)
    logger.info(f"Per-sample CSV → {csv_path}")

    # aggregate.json
    agg = {}
    for key in metric_keys:
        values = [m[key] for m in all_metrics]
        agg[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
        }
    agg["num_samples"] = success_count
    agg["config"] = {
        "dataset_root": args.dataset_root,
        "jax_server": args.jax_server,
        "torch_server": args.torch_server,
        "frame_count": args.frame_count,
        "action_horizon": ACTION_HORIZON,
        "action_dim": ACTION_DIM,
    }
    agg_path = output_dir / "aggregate.json"
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    logger.info(f"Aggregate JSON → {agg_path}")

    # worst_samples.csv
    sorted_metrics = sorted(all_metrics, key=lambda m: m["mse"], reverse=True)
    worst_path = output_dir / "worst_samples.csv"
    with open(worst_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=existing_keys)
        writer.writeheader()
        writer.writerows(sorted_metrics[:20])
    logger.info(f"Worst samples CSV → {worst_path}")

    logger.info(f"\n{'=' * 60}")
    logger.info("Aggregate Metrics (JAX vs PyTorch):")
    logger.info(f"  MSE:               {agg['mse']['mean']:.6e} ± {agg['mse']['std']:.6e}")
    logger.info(f"  MAE:               {agg['mae']['mean']:.6e} ± {agg['mae']['std']:.6e}")
    logger.info(f"  Max Abs Error:     {agg['max_abs_error']['mean']:.6e}")
    logger.info(f"  Cosine Sim:        {agg['cosine_similarity_mean']['mean']:.6f}")
    logger.info(f"  Agreement @ 1e-3:  {agg['agreement_1e-3']['mean']:.4%}")
    logger.info(f"  Agreement @ 1e-4:  {agg['agreement_1e-4']['mean']:.4%}")


if __name__ == "__main__":
    main()
