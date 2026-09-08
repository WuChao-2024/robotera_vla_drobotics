#!/usr/bin/env python3
"""Finalize a robotera_vla operator run according to the droboticflow contract.

Writes result.json (logs MUST stay an empty array so the platform keeps
collecting stdout), stage-manifest.json, prints the
DROBOTICFLOW_RESULT_PATH line, and reports artifacts with globally unique
ids of the form <portId>-<nodeId> (the worker dedupes by artifact id).

Modes:
  train       copy <step>/{params,assets} of the latest checkpoint to
              $OUTPUT_DIR/model (same layout as the reference
              M7_pickplace_example_ckpt) and report port "model".
  norm_stats  report work/assets/<config> on port "assets".
"""
import argparse
import json
import os
import pathlib
import shutil
import sys


def dir_size_bytes(path: pathlib.Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def artifact(node_id: str, port_id: str, name: str, path: pathlib.Path) -> dict:
    return {
        "id": f"{port_id}-{node_id}",
        "nodeId": node_id,
        "portId": port_id,
        "portName": port_id,
        "name": name,
        "type": "file",
        "path": str(path),
        "sizeBytes": dir_size_bytes(path),
    }


def finalize_train(ckpt_root: pathlib.Path, output_dir: pathlib.Path, config_name: str, exp_name: str, node_id: str) -> list:
    ckpt_root = ckpt_root / config_name / exp_name
    if not ckpt_root.is_dir():
        raise RuntimeError(f"checkpoint root not found: {ckpt_root}")
    steps = sorted(
        (int(p.name) for p in ckpt_root.iterdir() if p.is_dir() and p.name.isdigit())
    )
    if not steps:
        raise RuntimeError(f"no step checkpoints under {ckpt_root}")
    latest = ckpt_root / str(steps[-1])
    for required in ("params", "assets"):
        if not (latest / required).is_dir():
            raise RuntimeError(f"checkpoint {latest} missing '{required}'")

    model_dir = output_dir / "model"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True)
    shutil.copytree(latest / "params", model_dir / "params")
    shutil.copytree(latest / "assets", model_dir / "assets")

    print(
        f"[finalize] checkpoint step={steps[-1]} -> {model_dir} "
        f"({dir_size_bytes(model_dir)} bytes)"
    )
    return [artifact(node_id, "model", "M7 pi0.5 checkpoint", model_dir)]


def finalize_norm_stats(workdir: pathlib.Path, config_name: str, node_id: str) -> list:
    assets_dir = workdir / "assets" / config_name
    if not (assets_dir / "norm_stats.json").is_file():
        raise RuntimeError(f"norm_stats.json not found under {assets_dir}")
    print(f"[finalize] norm stats at {assets_dir}")
    return [artifact(node_id, "assets", "norm stats (assets)", assets_dir)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "norm_stats"])
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--ckpt-root", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-name", default="pi05_M7_pp_opensource")
    parser.add_argument("--exp-name", default="")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir).resolve()
    workdir = pathlib.Path(args.workdir).resolve()
    node_id = os.environ.get("NODE_ID", "node")

    if args.mode == "train":
        exp_name = args.exp_name or os.environ.get("RUN_ID", "exp")
        ckpt_root = pathlib.Path(args.ckpt_root).resolve() if args.ckpt_root else workdir / "checkpoints"
        artifacts = finalize_train(ckpt_root, output_dir, args.config_name, exp_name, node_id)
    else:
        artifacts = finalize_norm_stats(workdir, args.config_name, node_id)

    result = {"logs": [], "artifacts": artifacts, "resourceSeries": []}
    result_path = os.environ.get("RESULT_OUTPUT_PATH", "")
    manifest_path = os.environ.get("STAGE_MANIFEST_PATH", "")
    if not result_path or not manifest_path:
        print("ERROR: RESULT_OUTPUT_PATH / STAGE_MANIFEST_PATH not set", file=sys.stderr)
        return 1

    pathlib.Path(result_path).write_text(json.dumps(result, ensure_ascii=False))
    pathlib.Path(manifest_path).write_text('{"dataRefs": []}')
    print(f"DROBOTICFLOW_RESULT_PATH={result_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
