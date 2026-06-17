#!/usr/bin/env python3
"""
pyCauchyKesai vs hrt_model_exec (hbm_runtime) 脏数据推理性能对比。

每个 HBM 模型用两种接口各跑 10 帧（单线程）。
**全部通过独立子进程隔离**，避免 CauchyKesai 同进程多模型加载
导致 IOVA 冲突或 segfault。

用法:
  conda activate robotrea_python_runtime
  python bench_pycauchykesai_vs_hbmrt.py
"""

import glob
import json
import os
import re
import subprocess
import sys
import time

HBM_DIR = "/root/ssd/OELLM_Runtime/robotrea_model/v1"
FRAMES = 10
HRT_MODEL_EXEC = "hrt_model_exec"

# ---- pyCauchyKesai 子进程脚本 ----
# 每个模型在独立子进程中加载→推理→输出 JSON 结果
PYCK_WORKER_SCRIPT = r"""
import json, os, sys, time
os.environ["HB_DNN_USER_DEFINED_L2M_SIZES"] = "6:6:6:6"
import numpy as np
from pyCauchyKesai import CauchyKesai

hbm_path, cores_str = sys.argv[1], sys.argv[2]
n_cores = int(cores_str)

model = CauchyKesai(hbm_path, n_task=1, model_cnt_select=0)
info = model.s()
if n_cores > 1:
    model.set_scheduling_params(bpu_cores=list(range(n_cores)))

NP_DTYPE = {"float16": np.float16, "float32": np.float32,
            "int32": np.int32, "int64": np.int64}
inputs = []
for inp in info["inputs"]:
    shape = tuple(inp["shape"])
    dtype = NP_DTYPE[inp["dtype"]]
    if inp["dtype"] in ("float16", "float32"):
        data = np.random.randn(*shape).astype(dtype)
    else:
        ii = np.iinfo(dtype)
        data = np.random.randint(ii.min // 2, ii.max // 2, shape, dtype=dtype)
    inputs.append(np.ascontiguousarray(data))

total_mb = sum(x.nbytes for x in inputs) / (1024 * 1024)

# warmup 1x
model.inference(inputs)

# benchmark 10
wall_times = []; bpu_times = []
for _ in range(10):
    t0 = time.perf_counter()
    model.inference(inputs)
    t1 = time.perf_counter()
    wall_times.append((t1 - t0) * 1000)
    bpu_times.append(model.t()["time_ms"])

print("PYCK_OK:" + json.dumps({
    "avg_wall_ms": round(float(np.mean(wall_times)), 3),
    "min_wall_ms": round(float(np.min(wall_times)), 3),
    "max_wall_ms": round(float(np.max(wall_times)), 3),
    "avg_bpu_ms":  round(float(np.mean(bpu_times)), 3),
    "input_mb":    round(total_mb, 2),
}))
"""

# =============================================================================
# Test definitions
# =============================================================================
TESTS = [
    # (keyword, hrt_model_name, hrt_cores, prefer_suffix, pyck_cores)
    ("siglip",       "siglip",        "1",       "",                             1),
    ("gemma_expert", "gemma_expert",  "1,2,3,4", "",                             4),
    ("gemma_llm",    "gemma",         "1,2,3,4", "",                             4),
    ("gemma_llm",    "gemma",         "1,2,3,4", "hbdk20260318_EnableHpcTrue",   4),
    ("gemma_llm",    "gemma",         "1,2,3,4", "hbdk20260331_EnableHpcTrue",   4),
    ("gemma_llm",    "gemma",         "1,2,3,4", "hbdk20260318_EnableHpcFlase",  4),
]


def find_hbm(keyword: str, prefer: str) -> str:
    files = sorted(glob.glob(os.path.join(HBM_DIR, f"*{keyword}*.hbm")))
    files = [f for f in files if f.endswith(".hbm") and not f.endswith(".hbm.1")]
    if not files:
        raise FileNotFoundError(f"No .hbm matching '{keyword}'")
    if prefer:
        matches = [f for f in files if prefer in os.path.basename(f)]
        if matches:
            return matches[0]
    files.sort(key=lambda x: len(os.path.basename(x)))
    return files[0]


def make_label(hbm_path: str) -> str:
    name = os.path.basename(hbm_path)
    for old, new in [
        ("pi05_gemma_llm_pi05_action_horizon_20_ptq", "LLM"),
        ("pi05_gemma_expert_pi05_action_horizon_20_ptq", "Expert"),
        ("pi05_siglip_pi05_action_horizon_20_ptq", "SigLIP"),
        ("hbdk20260", "hbdk"),
        ("_EnableHpcTrue", "_HpcT"),
        ("_EnableHpcFlase", "_HpcF"),
        (".hbm", ""),
        ("__", "_"),
    ]:
        name = name.replace(old, new)
    name = name.strip("_")
    if name in ("LLM", ""):
        name = "LLM_base"
    return name


def run_hrt_perf(hbm_path: str, model_name: str, cores: str, label: str) -> dict:
    """hrt_model_exec perf 子进程."""
    cmd = [
        HRT_MODEL_EXEC, "perf",
        "--model_file", hbm_path,
        "--model_name", model_name,
        "--core_id", cores,
        "--thread_num", "1",
        "--frame_count", str(FRAMES),
    ]
    print(f"    [hrt] {label:<22s} ...", end=" ", flush=True)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
            env={**os.environ, "HB_DNN_USER_DEFINED_L2M_SIZES": "6:6:6:6"})
        combined = proc.stderr + proc.stdout

        avg_ms = None; fps = None
        for pat, key in [(r"Average\s+latency\s+is:\s+([\d.]+)\s+ms", "avg"),
                          (r"Frame\s+rate\s+is:\s+([\d.]+)\s+FPS", "fps")]:
            m = re.search(pat, combined)
            if m and key == "avg": avg_ms = float(m.group(1))
            if m and key == "fps": fps = float(m.group(1))

        if avg_ms is None:
            err_lines = [l.strip() for l in combined.splitlines()
                         if "[E]" in l or "error" in l.lower()]
            err = err_lines[0][:100] if err_lines else "unknown"
            print(f"FAIL")
            return {"label": label, "avg_wall_ms": None, "error": err}

        print(f"avg={avg_ms:.2f}ms  fps={fps:.1f}")
        return {"label": label, "avg_wall_ms": round(avg_ms, 3),
                "fps": round(fps, 3) if fps else None}
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return {"label": label, "avg_wall_ms": None, "error": "timeout"}


def run_pyck_subprocess(hbm_path: str, cores: int, label: str) -> dict:
    """pyCauchyKesai 子进程隔离推理."""
    print(f"    [pyCK] {label:<22s} ...", end=" ", flush=True)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", PYCK_WORKER_SCRIPT, hbm_path, str(cores)],
            capture_output=True, text=True, timeout=300,
            env={**os.environ, "HB_DNN_USER_DEFINED_L2M_SIZES": "6:6:6:6"},
        )
        for line in proc.stdout.splitlines():
            if line.startswith("PYCK_OK:"):
                result = json.loads(line[len("PYCK_OK:"):])
                result["label"] = label
                aw = result["avg_wall_ms"]
                ab = result["avg_bpu_ms"]
                print(f"wall={aw:.2f}ms  bpu={ab:.2f}ms  "
                      f"({result['min_wall_ms']:.2f}~{result['max_wall_ms']:.2f})")
                return result

        # failed
        err = (proc.stderr + proc.stdout)[-300:]
        err_line = ""
        for line in err.splitlines():
            if "[E]" in line or "Error" in line or "FAIL" in line:
                err_line = line.strip()[:120]
                break
        print(f"FAIL")
        return {"label": label, "avg_wall_ms": None, "error": err_line or err[:120]}
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return {"label": label, "avg_wall_ms": None, "error": "timeout"}


# =============================================================================
# main
# =============================================================================
def main():
    print("=" * 70)
    print("pyCauchyKesai vs hrt_model_exec (hbm_runtime) 脏数据推理对比")
    print(f"帧数: {FRAMES}, 单线程, L2 Cache=6:6:6:6")
    print(f"隔离: 每个模型每次推理均为独立子进程")
    print("=" * 70)

    pyck_results = []
    hrt_results = []

    for keyword, model_name, cores, prefer, pyck_cores in TESTS:
        try:
            hbm = find_hbm(keyword, prefer)
        except FileNotFoundError as e:
            print(f"\n  SKIP: {e}")
            continue

        label = make_label(hbm)
        print(f"\n{'─'*60}")
        print(f"  [{label}] {os.path.basename(hbm)}")
        print(f"{'─'*60}")

        hrt = run_hrt_perf(hbm, model_name, cores, label)
        hrt_results.append(hrt)
        time.sleep(0.3)

        pyck = run_pyck_subprocess(hbm, pyck_cores, label)
        pyck_results.append(pyck)
        time.sleep(0.3)

    # ---- Print comparison ----
    print("\n")
    print("=" * 88)
    print("                           结 果 对 比")
    print("=" * 88)
    hdr = (f"{'模型':<30s} | {'pyCK wall':>10s} | "
           f"{'hrt_model_exec':>14s} | {'diff':>8s} | {'overhead':>8s}")
    print(hdr)
    print("-" * 88)

    for pyck, hrt in zip(pyck_results, hrt_results):
        a = pyck.get("avg_wall_ms")
        b = hrt.get("avg_wall_ms")
        if a is None or b is None:
            e1 = pyck.get("error", "")[:25] if a is None else ""
            e2 = hrt.get("error", "")[:25] if b is None else ""
            print(f"{pyck['label']:<30s} | {'FAIL':>10s} | {'FAIL':>14s} | {'—':>8s} | {'—':>8s}")
            continue
        diff = a - b
        oh = (diff / b) * 100
        fps_s = f"{hrt.get('fps', 0):.1f}"
        print(f"{pyck['label']:<30s} | {a:8.2f}ms | "
              f"{b:8.2f}ms ({fps_s:>5s}) | {diff:+7.2f}ms | {oh:+6.1f}%")

    print("-" * 88)
    print(f"\n  {'pyCK = pyCauchyKesai Python API':<45s} (独立子进程, 脏数据, 10 iters avg wall)")
    print(f"  {'hrt  = hrt_model_exec perf':<45s} (独立子进程, --thread_num 1 --frame_count 10)")
    print(f"  {'diff = pyCK wall - hrt avg':<45s} (正值 = Python 接口 overhead)")
    print(f"  {'overhead = diff / hrt * 100%':<45s}")
    print(f"  {'pyCK wall 含: Python调度 + numpy→ION拷贝 + BPU提交等待':<60s}")
    print(f"  {'pyCK bpu  = CauchyKesai.t() 纯BPU推理时间':<60s}")
    print(f"  {'hrt  avg  = hrt_model_exec perf avg latency':<60s}")

    # ---- pyCK detail ----
    print("\n")
    print("=" * 78)
    print("pyCauchyKesai 子进程详细统计")
    print("=" * 78)
    hdr2 = (f"{'模型':<30s} | {'wall avg':>9s} | {'wall min':>9s} | "
            f"{'wall max':>9s} | {'bpu avg':>9s} | {'input':>7s}")
    print(hdr2)
    print("-" * 78)
    for r in pyck_results:
        if r.get("avg_wall_ms") is None:
            print(f"{r['label']:<30s} | {'FAIL':>9s} | {'—':>9s} | {'—':>9s} | {'—':>9s} | {'—':>7s}")
            continue
        print(f"{r['label']:<30s} | {r['avg_wall_ms']:7.2f}ms | "
              f"{r['min_wall_ms']:7.2f}ms | {r['max_wall_ms']:7.2f}ms | "
              f"{r['avg_bpu_ms']:7.2f}ms | {r['input_mb']:5.1f}M")


if __name__ == "__main__":
    main()
