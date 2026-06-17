#!/usr/bin/env python3
"""pyCauchyKesai vs hbm_runtime vs hrt_model_exec perf — 三层对比, 串行."""

import glob, os, re, subprocess, sys, tempfile, time
import numpy as np

os.environ["HB_DNN_USER_DEFINED_L2M_SIZES"] = "6:6:6:6"

from pyCauchyKesai import CauchyKesai

HBM_DIR = "/root/ssd/OELLM_Runtime/robotrea_model/v1"
FRAMES = 10
DT = {"float16": np.float16, "float32": np.float32, "int32": np.int32, "int64": np.int64}

TESTS = [
    ("gemma_llm",    "gemma",         "1,2,3,4", 4, ""),
    ("gemma_llm",    "gemma",         "1,2,3,4", 4, "hbdk20260318_EnableHpcTrue"),
    ("gemma_llm",    "gemma",         "1,2,3,4", 4, "hbdk20260331_EnableHpcTrue"),
    ("gemma_llm",    "gemma",         "1,2,3,4", 4, "hbdk20260318_EnableHpcFlase"),
    ("siglip",       "siglip",        "1,2,3,4", 1, ""),
    ("gemma_expert", "gemma_expert",  "1,2,3,4", 4, ""),
]


def find_hbm(kw, prefer=""):
    files = sorted(glob.glob(f"{HBM_DIR}/*{kw}*.hbm"))
    files = [f for f in files if f.endswith(".hbm") and not f.endswith(".hbm.1")]
    if not files: raise FileNotFoundError(kw)
    if prefer:
        m = [f for f in files if prefer in os.path.basename(f)]
        if m: return m[0]
    files.sort(key=lambda x: len(os.path.basename(x)))
    return files[0]


def label(hbm):
    n = os.path.basename(hbm)
    for o, r in [("pi05_gemma_llm_pi05_action_horizon_20_ptq","LLM"),
                 ("pi05_gemma_expert_pi05_action_horizon_20_ptq","Expert"),
                 ("pi05_siglip_pi05_action_horizon_20_ptq","SigLIP"),
                 ("hbdk20260","hbdk"),("_EnableHpcTrue","_HpcT"),
                 ("_EnableHpcFlase","_HpcF"),(".hbm",""),("__","_")]:
        n = n.replace(o, r)
    n = n.strip("_")
    return "LLM_base" if n in ("LLM","") else n


def run_hrt_model_exec(hbm, mn, cores, input_bins=None):
    """Run hrt_model_exec perf. Returns (avg_ms, fps) or (None, None)."""
    cmd = ["hrt_model_exec", "perf", "--model_file", hbm, "--model_name", mn,
           "--core_id", cores, "--thread_num", "1", "--frame_count", str(FRAMES)]
    if input_bins:
        cmd += ["--input_file", ",".join(input_bins)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    c = proc.stderr + proc.stdout
    avg_ms = float(m.group(1)) if (m := re.search(r"Average\s+latency\s+is:\s+([\d.]+)\s+ms", c)) else None
    fps = float(m.group(1)) if (m := re.search(r"Frame\s+rate\s+is:\s+([\d.]+)\s+FPS", c)) else None
    if avg_ms is None:
        err = [l.strip() for l in c.splitlines() if "[E]" in l]
        if err: print(f"      hrt err: {err[0][:100]}")
    return avg_ms, fps


def gen_dirty_inputs(model):
    """Generate random numpy inputs matching model spec."""
    info = model.s()
    ins = []
    for inp in info["inputs"]:
        shape = tuple(inp["shape"]); dtype = DT[inp["dtype"]]
        if inp["dtype"] in ("float16","float32"):
            data = np.random.randn(*shape).astype(dtype)
        else:
            ii = np.iinfo(dtype); data = np.random.randint(ii.min//2, ii.max//2, shape, dtype=dtype)
        ins.append(np.ascontiguousarray(data))
    return ins


# =============================================================================
# Main
# =============================================================================
print("=" * 78)
print("  pyCauchyKesai vs hbm_runtime vs hrt_model_exec perf — 三层对比")
print(f"  帧数: {FRAMES}, 单线程, 各自独立随机脏数据")
print("=" * 78)

results = []

for kw, mn, cores, ncores, pref in TESTS:
    try:
        hbm = find_hbm(kw, pref)
    except FileNotFoundError:
        print(f"\n  SKIP: {kw}/{pref}")
        continue
    lb = label(hbm)
    print(f"\n{'─'*60}\n  [{lb}] {os.path.basename(hbm)}\n{'─'*60}")

    l1_wall = l1_bpu = l2_avg = l3_avg = None

    # ── Load model once for L1+L2 ──
    try:
        print(f"  [L1+L2] loading ...", end=" ", flush=True)
        model = CauchyKesai(hbm, n_task=1, model_cnt_select=0)
        if ncores > 1:
            model.set_scheduling_params(bpu_cores=list(range(ncores)))
        inputs = gen_dirty_inputs(model)

        # ── L1: pyCauchyKesai ──
        model.inference(inputs)  # warmup
        ws, bs = [], []
        for _ in range(FRAMES):
            t0 = time.perf_counter()
            model.inference(inputs)
            t1 = time.perf_counter()
            ws.append((t1-t0)*1000)
            bs.append(model.t()["time_ms"])
        l1_wall = float(np.mean(ws)); l1_bpu = float(np.mean(bs))
        print(f"L1-pyCK wall={l1_wall:.2f}ms bpu={l1_bpu:.2f}ms", end="  ")

        # ── Save bins, release model ──
        with tempfile.TemporaryDirectory() as td:
            bin_paths = []
            for i, arr in enumerate(inputs):
                p = f"{td}/in_{i}.bin"; arr.tofile(p); bin_paths.append(p)
            del model

            # ── L2: hbm_runtime (perf + same bins) ──
            l2_avg, l2_fps = run_hrt_model_exec(hbm, mn, cores, bin_paths)
            if l2_avg:
                print(f"L2-hbmRT={l2_avg:.2f}ms")
            else:
                print(f"L2-hbmRT=FAIL")
    except Exception as e:
        print(f"FAIL: {e}")
        try: del model
        except: pass

    # ── L3: hrt_model_exec perf (auto data, no input) ──
    l3_avg, l3_fps = run_hrt_model_exec(hbm, mn, cores, None)
    if l3_avg:
        print(f"  [L3-hrtP] auto data  avg={l3_avg:.2f}ms  fps={l3_fps:.1f}")
    else:
        print(f"  [L3-hrtP] FAIL")

    results.append((lb, l1_wall, l1_bpu, l2_avg, l3_avg))
    time.sleep(0.3)

# ── Table ──
print("\n\n" + "=" * 100)
print("                        三 层 对 比 结 果")
print("=" * 100)
hdr = (f"{'模型':<26s} | {'L1 pyCK wall':>12s} | {'L2 hbmRT bin':>13s} | "
       f"{'L3 hrtP auto':>13s} | {'L1-L2':>8s} | {'L1-L3':>8s} | {'L2-L3':>8s}")
print(hdr); print("-" * 100)
for lb, l1w, l1b, l2a, l3a in results:
    ok = all(x is not None for x in [l1w, l2a, l3a])
    if not ok:
        def s(x): return f"{x:.2f}ms" if x else "FAIL"
        print(f"{lb:<26s} | {s(l1w):>12s} | {s(l2a):>13s} | {s(l3a):>13s}")
        continue
    print(f"{lb:<26s} | {l1w:10.2f}ms | {l2a:11.2f}ms | {l3a:11.2f}ms | "
          f"{l1w-l2a:+6.2f}ms | {l1w-l3a:+6.2f}ms | {l2a-l3a:+6.2f}ms")
print("-" * 100)
print("""
  L1 pyCK wall = pyCauchyKesai.inference() perf_counter 端到端 (Python全栈)
  L2 hbmRT bin = hrt_model_exec perf --input_file (C++原生, 相同脏.bin)
  L3 hrtP auto = hrt_model_exec perf 无输入 (C++原生, 工具自生成数据)

  L1-L2 = Python overhead (调度+numpy→ION拷贝+BPU提交等待)
  L2-L3 = 同运行时, 输入方式差异 (.bin文件 vs 内部自生成)
""")

print("=" * 78)
print(" pyCauchyKesai wall vs bpu (Python调度+拷贝开销 vs 纯BPU时间)")
print("=" * 78)
for lb, l1w, l1b, l2a, l3a in results:
    if l1w is None: continue
    print(f"  {lb:<26s} wall={l1w:7.2f}ms  bpu={l1b:7.2f}ms  wall-bpu={l1w-l1b:+6.2f}ms")
