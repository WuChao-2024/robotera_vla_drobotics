#!/usr/bin/env python3
"""
BC vs Torch 批量并发对比。

N 个 BC server 各处理 1 帧，使用线程池并发请求。
Torch server 串行推理（很快）。

用法:
  python compare_batch.py --torch-port 8004 --bc-ports 9200,9201,...,9249 --frames-per-server 1
"""

import argparse
import csv
import io
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import zmq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bc_batch_compare")

# =============================================================================
# 数据集加载
# =============================================================================

DEFAULT_DATASET_ROOT = "/home/chao.wu/pi_oellm/ROBOTREA/M7_pickplace_example_merged_LeRobotDataSetv2.1"


def load_tasks(dataset_root):
    tasks_path = os.path.join(dataset_root, "meta", "tasks.jsonl")
    tasks = {}
    with open(tasks_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            tasks[item["task_index"]] = item["task"]
    return tasks


def load_frames(dataset_root, total_frames):
    import pyarrow.parquet as pq
    import cv2

    data_dir = os.path.join(dataset_root, "data", "chunk-000")
    episodes = sorted(
        [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".parquet")]
    )
    logger.info(f"Found {len(episodes)} episodes")

    video_dir = os.path.join(dataset_root, "videos", "chunk-000")
    cam_keys = ["observation.images.cam_high", "observation.images.cam_left", "observation.images.cam_right"]
    output_keys = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

    frames = []
    for ep_path in episodes:
        if len(frames) >= total_frames:
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
            if len(frames) >= total_frames:
                break
            row = df.iloc[row_idx]
            frame_idx = int(row["frame_index"])

            images = {}
            decode_ok = True
            for out_key, cap in caps.items():
                cap.set(1, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    decode_ok = False
                    break
                images[out_key] = frame[..., ::-1]

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
# 同步 ZMQ 客户端（线程安全，每个线程独立 Context）
# =============================================================================

def sync_infer(server_addr, images, state, text, timeout_ms=600000):
    """同步 ZMQ 推理请求。每个调用创建独立 Context，线程安全。"""
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(server_addr)

    npz_data = {"images": images, "state": state.astype(np.float32), "text": text}
    buf = io.BytesIO()
    np.savez_compressed(buf, **npz_data)

    t0 = time.time()
    socket.send(buf.getvalue())
    reply = socket.recv_pyobj()
    elapsed = time.time() - t0

    actions = np.squeeze(reply["actions"])
    socket.close()
    ctx.term()
    return actions, elapsed


# =============================================================================
# 指标计算
# =============================================================================

def compute_metrics(a_bc, a_torch, ground_truth=None):
    diff = a_bc - a_torch
    abs_diff = np.abs(diff)

    mse = float(np.mean(diff ** 2))
    cos_sims = []
    for t in range(a_bc.shape[0]):
        dot = np.dot(a_bc[t], a_torch[t])
        norm = np.linalg.norm(a_bc[t]) * np.linalg.norm(a_torch[t]) + 1e-8
        cos_sims.append(float(dot / norm))

    metrics = {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(abs_diff)),
        "max_abs_error": float(np.max(abs_diff)),
        "cosine_similarity_mean": float(np.mean(cos_sims)),
        "agreement_1e-2": float(np.mean(abs_diff < 1e-2)),
        "agreement_1e-3": float(np.mean(abs_diff < 1e-3)),
    }

    if ground_truth is not None:
        if ground_truth.ndim == 1:
            gt = ground_truth[np.newaxis, :]
        else:
            gt = ground_truth[:a_bc.shape[0], :a_bc.shape[1]]
        metrics["mse_bc_gt"] = float(np.mean((a_bc[:gt.shape[0], :gt.shape[1]] - gt) ** 2))
        metrics["mse_torch_gt"] = float(np.mean((a_torch[:gt.shape[0], :gt.shape[1]] - gt) ** 2))

    return metrics


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="BC vs Torch 批量并发对比")
    p.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--torch-port", type=int, default=8004)
    p.add_argument("--bc-ports", type=str, required=True,
                   help="逗号分隔的 BC server 端口列表")
    p.add_argument("--frames-per-server", type=int, default=1,
                   help="每个 BC server 处理的帧数")
    p.add_argument("--output-dir", type=str, default="./batch_comparison_results")
    return p.parse_args()


def run_comparison(args):
    bc_ports = [int(p.strip()) for p in args.bc_ports.split(",")]
    num_bc_servers = len(bc_ports)
    total_frames = num_bc_servers * args.frames_per_server

    logger.info("=" * 60)
    logger.info("BC vs Torch 批量并发对比 (ThreadPoolExecutor)")
    logger.info(f"  Dataset:         {args.dataset_root}")
    logger.info(f"  Torch server:    tcp://localhost:{args.torch_port}")
    logger.info(f"  BC servers:      {num_bc_servers} (ports {bc_ports[0]}-{bc_ports[-1]})")
    logger.info(f"  Frames/server:   {args.frames_per_server}")
    logger.info(f"  Total frames:    {total_frames}")
    logger.info(f"  Output:          {args.output_dir}")
    logger.info("=" * 60)

    # 1. 加载数据
    tasks = load_tasks(args.dataset_root)
    frames = load_frames(args.dataset_root, total_frames)
    if len(frames) < total_frames:
        logger.warning(f"Only loaded {len(frames)} frames, requested {total_frames}")
        total_frames = len(frames)

    # 分配帧到 BC server
    server_assignments = []
    for i, port in enumerate(bc_ports):
        start = i * args.frames_per_server
        end = min(start + args.frames_per_server, total_frames)
        server_assignments.append((port, list(range(start, end))))

    # 2. 逐帧 Torch 推理（串行）
    logger.info("Running Torch inference (serial)...")
    torch_addr = f"tcp://localhost:{args.torch_port}"
    torch_results = {}
    torch_total_time = 0.0

    for idx in range(total_frames):
        frame = frames[idx]
        actions, elapsed = sync_infer(
            torch_addr, frame["images"], frame["state"], frame["prompt"],
            timeout_ms=60000,
        )
        torch_results[idx] = actions
        torch_total_time += elapsed

    logger.info(f"Torch done: {total_frames} frames in {torch_total_time:.1f}s "
                f"({torch_total_time/total_frames:.3f}s/frame)")

    # 3. 并发 BC 推理 — 线程池，每个线程独立 zmq Context
    logger.info(f"Running BC inference ({num_bc_servers} servers, "
                f"{args.frames_per_server} frames each, ThreadPool)...")
    bc_addr_map = {port: f"tcp://localhost:{port}" for port in bc_ports}

    bc_results = {}

    def bc_infer_task(port, idx):
        addr = bc_addr_map[port]
        frame = frames[idx]
        try:
            actions, elapsed = sync_infer(addr, frame["images"], frame["state"], frame["prompt"],
                                           timeout_ms=600000)
            logger.info(f"  [port {port}] frame {idx}: {elapsed:.1f}s")
            return idx, actions, elapsed
        except Exception as e:
            logger.error(f"  [port {port}] frame {idx} FAILED: {e}")
            return idx, None, 0.0

    # 构建 task 列表：每个 server 串行处理分配帧，servers 之间并行
    # 每个 server 内的帧串行执行（BC server 是 REP 模式，只能逐个响应）
    t_bc_start = time.time()

    with ThreadPoolExecutor(max_workers=num_bc_servers) as pool:
        # 每个 server 是一个串行子任务
        futures = []
        for port, indices in server_assignments:
            # server 内串行，提交一个 chain 任务
            def server_chain(port_, indices_):
                results = []
                for idx_ in indices_:
                    r = bc_infer_task(port_, idx_)
                    results.append(r)
                return results
            futures.append(pool.submit(server_chain, port, indices))

        for future in as_completed(futures):
            chain_results = future.result()
            for idx, actions, elapsed in chain_results:
                bc_results[idx] = (actions, elapsed)

    bc_wall_time = time.time() - t_bc_start

    bc_success = sum(1 for v in bc_results.values() if v[0] is not None)
    logger.info(f"BC done: {bc_success}/{total_frames} frames, "
                f"wall time {bc_wall_time:.1f}s "
                f"(~{bc_wall_time/max(bc_success,1):.1f}s/frame equivalent)")

    # 4. 计算指标
    all_metrics = []
    for idx in range(total_frames):
        if idx not in torch_results or idx not in bc_results or bc_results[idx][0] is None:
            continue

        bc_actions, bc_time = bc_results[idx]
        torch_actions = torch_results[idx]

        m = compute_metrics(bc_actions, torch_actions, ground_truth=frames[idx]["ground_truth_action"])
        m["sample_index"] = idx
        m["episode_name"] = frames[idx]["meta"]["episode_name"]
        m["frame_index"] = int(frames[idx]["meta"]["frame_index"])
        m["prompt"] = frames[idx]["prompt"]
        m["bc_time_s"] = bc_time
        m["torch_time_s"] = torch_total_time / total_frames
        all_metrics.append(m)

    # 5. 输出
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not all_metrics:
        logger.error("No successful comparisons!")
        return

    metric_keys = ["mse", "rmse", "mae", "max_abs_error", "cosine_similarity_mean",
                   "agreement_1e-2", "agreement_1e-3"]
    extra_keys = ["mse_bc_gt", "mse_torch_gt", "bc_time_s", "torch_time_s"]
    all_keys = ["sample_index", "episode_name", "frame_index", "prompt"] + metric_keys + extra_keys
    existing_keys = [k for k in all_keys if k in all_metrics[0]]

    csv_path = output_dir / "per_sample.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=existing_keys)
        writer.writeheader()
        writer.writerows(all_metrics)
    logger.info(f"Per-sample CSV → {csv_path}")

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
    if "mse_bc_gt" in all_metrics[0]:
        for key in ["mse_bc_gt", "mse_torch_gt"]:
            values = [m[key] for m in all_metrics]
            agg[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
            }
    agg["num_samples"] = len(all_metrics)
    agg["config"] = {
        "num_bc_servers": num_bc_servers,
        "frames_per_server": args.frames_per_server,
        "total_frames": total_frames,
        "bc_wall_time_s": bc_wall_time,
        "torch_total_time_s": torch_total_time,
    }
    agg_path = output_dir / "aggregate.json"
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    logger.info(f"Aggregate JSON → {agg_path}")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"BC vs Torch 批量对比汇总 ({len(all_metrics)} samples)")
    logger.info(f"{'=' * 60}")
    logger.info(f"  MSE:               {agg['mse']['mean']:.6e} ± {agg['mse']['std']:.6e}")
    logger.info(f"  MAE:               {agg['mae']['mean']:.6e} ± {agg['mae']['std']:.6e}")
    logger.info(f"  Max Abs Error:     {agg['max_abs_error']['mean']:.6e}")
    logger.info(f"  Cosine Sim:        {agg['cosine_similarity_mean']['mean']:.6f}")
    logger.info(f"  Agreement @ 1e-2:  {agg['agreement_1e-2']['mean']:.4%}")
    logger.info(f"  Agreement @ 1e-3:  {agg['agreement_1e-3']['mean']:.4%}")
    if "mse_bc_gt" in agg:
        logger.info(f"  MSE BC-GT:         {agg['mse_bc_gt']['mean']:.6e}")
        logger.info(f"  MSE Torch-GT:      {agg['mse_torch_gt']['mean']:.6e}")
    logger.info(f"  BC wall time:      {bc_wall_time:.1f}s ({num_bc_servers} servers parallel)")
    logger.info(f"  Torch total time:  {torch_total_time:.1f}s (serial)")
    logger.info(f"{'=' * 60}")


def main():
    args = parse_args()
    run_comparison(args)


if __name__ == "__main__":
    main()