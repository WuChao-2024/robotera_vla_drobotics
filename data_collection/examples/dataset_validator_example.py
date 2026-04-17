#!/usr/bin/env python3
"""Example helper: validate minimal dataset structure (LeRobot v2.0 format).

Checks meta files, first chunk, and first episode's parquet/video existence.
No heavy dependencies are required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LOG_PREFIX = "[2026-04-13][dataset_validator_example:main]"

REQUIRED_INFO_FIELDS = {
    "codebase_version",
    "robot_type",
    "fps",
    "chunks_size",
    "total_chunks",
    "total_episodes",
    "total_frames",
    "total_tasks",
    "total_videos",
    "data_path",
    "video_path",
    "features",
    "splits",
}

REQUIRED_EPISODE_FIELDS = {"episode_index", "length", "tasks"}
REQUIRED_TASK_FIELDS = {"task", "task_index"}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def validate(dataset_root: Path) -> list[str]:
    errors: list[str] = []

    # --- meta/info.json ---
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        errors.append(f"missing file: {info_path}")
        return errors

    info = load_json(info_path)
    missing_info = REQUIRED_INFO_FIELDS - set(info.keys())
    if missing_info:
        errors.append(f"info.json missing fields: {sorted(missing_info)}")

    # --- meta/episodes.jsonl ---
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        errors.append(f"missing file: {episodes_path}")
    else:
        episodes = load_jsonl(episodes_path)
        if not episodes:
            errors.append("episodes.jsonl is empty")
        else:
            missing_ep = REQUIRED_EPISODE_FIELDS - set(episodes[0].keys())
            if missing_ep:
                errors.append(f"episodes.jsonl missing fields: {sorted(missing_ep)}")

    # --- meta/task.jsonl ---
    task_path = dataset_root / "meta" / "task.jsonl"
    if not task_path.exists():
        errors.append(f"missing file: {task_path}")
    else:
        tasks = load_jsonl(task_path)
        if not tasks:
            errors.append("task.jsonl is empty")
        else:
            missing_task = REQUIRED_TASK_FIELDS - set(tasks[0].keys())
            if missing_task:
                errors.append(f"task.jsonl missing fields: {sorted(missing_task)}")

    # --- data/chunk-000 and first episode parquet ---
    chunk_dir = dataset_root / "data" / "chunk-000"
    if not chunk_dir.exists():
        errors.append(f"missing dir: {chunk_dir}")
    else:
        parquet_files = sorted(chunk_dir.glob("episode_*.parquet"))
        if not parquet_files:
            errors.append(f"no parquet files found in {chunk_dir}")

    # --- videos: check first episode per video feature ---
    features = info.get("features", {})
    video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
    videos_chunk_dir = dataset_root / "videos" / "chunk-000"
    if not videos_chunk_dir.exists():
        errors.append(f"missing dir: {videos_chunk_dir}")
    else:
        for vk in video_keys:
            vk_dir = videos_chunk_dir / vk
            if not vk_dir.exists():
                errors.append(f"missing video dir: {vk_dir}")
                continue
            mp4_files = sorted(vk_dir.glob("episode_*.mp4"))
            if not mp4_files:
                errors.append(f"no mp4 files found in {vk_dir}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Robotera dataset structure (lightweight).")
    parser.add_argument("--dataset-root", required=True, help="Path to dataset root")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    errors = validate(dataset_root)

    if errors:
        print(f"{LOG_PREFIX} validation failed")
        for err in errors:
            print(f"- {err}")
        return 1

    print(f"{LOG_PREFIX} validation passed for {dataset_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
