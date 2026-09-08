#!/usr/bin/env python3
"""Resolve upstream artifacts (datasets + norm_stats assets) for robotera_vla operators.

Port contract (v1.1.0, assets-first redesign):
  - The single input port with id "assets" (declared first) carries the
    norm_stats.json produced by the stats operator.
  - EVERY other wired input port is one LeRobot collection root, regardless of
    its id: "dataset", "dataset_2", UI-generated UUIDs, any name. There is no
    naming rule and no count limit; the workflow editor forbids two edges into
    one port, so one port == one dataset.
  - Strict validation, no fallbacks:
      * a dataset port not pointing at a meta/ + data/ collection  -> hard error
      * a wired assets port without norm_stats.json                -> hard error
      * zero dataset ports                                        -> hard error

Inputs are read from INPUT_BINDINGS_PATH (platform JSON, grouped by this
node's input ports; authoritative) with INPUT_ARTIFACTS_PATH (flat artifact
list, portId = upstream output port names) as fallback.

Outputs (printed, consumed by entry scripts):
  DATASET_PORTS_FILE=<file>   one collection path per line
  ASSETS_READY=1              norm_stats.json from the "assets" port is in place
"""
import argparse
import json
import pathlib
import shutil
import sys

ASSETS_PORT_ID = "assets"


def to_local_path(raw_path: str) -> pathlib.Path | None:
    """Map an artifact path to a local readable path.

    bos://testxyz/<key> maps onto the workspace PVC mount (csi-bos mounts the
    bucket root at /workspace). Other buckets are not mounted locally.
    """
    if raw_path.startswith("bos://"):
        rest = raw_path[len("bos://"):]
        bucket, _, key = rest.partition("/")
        if bucket == "testxyz" and key.strip("/"):
            return pathlib.Path("/workspace") / key.strip("/")
        return None
    return pathlib.Path(raw_path)


def load_json(path_str: str):
    p = pathlib.Path(path_str)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text() or "[]")
    except json.JSONDecodeError as exc:
        print(f"ERROR: cannot parse {p}: {exc}", file=sys.stderr)
        raise SystemExit(1)


def collect_ports(bindings, artifacts):
    """Return (dataset_entries, assets_entries) as [(port_id, raw_path)].

    Bindings are preferred (targetPortId = this node's input port, the
    authoritative wiring). Every wired port other than "assets" is a dataset
    port; port ids carry no other semantics. All artifacts of a binding are
    collected (the editor normally enforces one edge per port).
    """
    dataset_entries, assets_entries = [], []

    def take(target, raw):
        if target == ASSETS_PORT_ID:
            assets_entries.append((target, raw))
        elif target:
            dataset_entries.append((target, raw))

    if bindings:
        for binding in bindings if isinstance(bindings, list) else []:
            if not isinstance(binding, dict):
                continue
            target = str(binding.get("targetPortId", "")).strip()
            for art in binding.get("artifacts") or []:
                raw = str(art.get("path", "")).strip()
                if raw:
                    take(target, raw)
        if dataset_entries or assets_entries:
            return dataset_entries, assets_entries
    # fallback: flat artifacts, portId is the upstream OUTPUT port name
    for art in artifacts or []:
        if not isinstance(art, dict):
            continue
        raw = str(art.get("path", "")).strip()
        if raw:
            take(str(art.get("portId", "")).strip(), raw)
    return dataset_entries, assets_entries


def is_lerobot_collection(path: pathlib.Path) -> bool:
    return (path / "meta").is_dir() and (path / "data").is_dir()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--config-name", default="pi05_M7_pp_opensource")
    parser.add_argument("--input-artifacts-path", required=True)
    parser.add_argument("--input-bindings-path", default="")
    parser.add_argument("--dataset-ports-file", required=True)
    args = parser.parse_args()

    artifacts = load_json(args.input_artifacts_path)
    bindings = load_json(args.input_bindings_path) if args.input_bindings_path else None
    if artifacts is None and bindings is None:
        print("[prepare_inputs] no upstream artifacts/bindings")

    dataset_entries, assets_entries = collect_ports(bindings, artifacts)

    # ---- dataset ports: every wired non-assets port, strictly validated ----
    repos, seen = [], set()
    for port_id, raw_path in dataset_entries:
        path = to_local_path(raw_path)
        if path is None:
            print(
                f"ERROR: dataset port '{port_id}' points at an unmapped remote path: "
                f"{raw_path} (only bos://testxyz/<key>, mounted at /workspace, "
                f"is readable without download)",
                file=sys.stderr,
            )
            return 1
        if not path.is_dir():
            print(
                f"ERROR: dataset port '{port_id}' path not accessible locally: {raw_path}",
                file=sys.stderr,
            )
            return 1
        if not is_lerobot_collection(path):
            print(
                f"ERROR: dataset port '{port_id}' is not a LeRobot collection "
                f"(needs meta/ + data/): {raw_path}\n"
                f"       数据集端口必须指向单个 LeRobot 数据集根目录; 请修正连线指向, "
                f"或在前面增加格式转换算子",
                file=sys.stderr,
            )
            return 1
        resolved = str(path.resolve())
        if resolved in seen:
            print(f"[prepare_inputs] WARNING: duplicate dataset path skipped (port '{port_id}'): {resolved}")
            continue
        seen.add(resolved)
        repos.append(resolved)

    if not repos:
        print(
            "ERROR: no dataset ports wired; datasets can only enter via port wiring "
            "(目录读取 -> 任意 dataset 输入端口连线)",
            file=sys.stderr,
        )
        return 1

    repos.sort()  # bindings 顺序不稳定, 按路径排序保证 data_config 可复现
    ports_file = pathlib.Path(args.dataset_ports_file)
    ports_file.parent.mkdir(parents=True, exist_ok=True)
    ports_file.write_text("\n".join(repos) + "\n")
    print(f"DATASET_PORTS_FILE={ports_file}")
    print(f"[prepare_inputs] dataset ports wired: {len(repos)}")
    for repo in repos:
        print(f"[prepare_inputs]   {repo}")

    # ---- assets port: norm_stats.json, wired-but-missing is fatal ----
    if assets_entries:
        src = None
        searched = []
        for port_id, raw_path in assets_entries:
            base = to_local_path(raw_path)
            if base is None:
                searched.append(f"{raw_path} (unmapped remote path)")
                continue
            for cand in (base, base / args.config_name):
                if (cand / "norm_stats.json").is_file():
                    src = cand / "norm_stats.json"
                    break
            searched.append(str(base))
            if src is not None:
                break
        if src is None:
            print(
                f"ERROR: assets port wired but norm_stats.json not found under: "
                f"{'; '.join(searched)}\n"
                f"       assets 端口(第一输入)必须连『M7 pi0.5 归一化统计』的 assets 输出; "
                f"若想让训练自动补算统计, 请勿连接 assets 端口",
                file=sys.stderr,
            )
            return 1
        dst_dir = pathlib.Path(args.workdir) / "assets" / args.config_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / "norm_stats.json")
        print(f"[prepare_inputs] copied {src} -> {dst_dir / 'norm_stats.json'}")
        print("ASSETS_READY=1")
    else:
        print("ASSETS_READY=0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
