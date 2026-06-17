#!/usr/bin/env python3
"""
pyCauchyKesai vs hbm_runtime vs hrt_model_exec perf 三层推理性能对比。
串行执行，每模型依次跑 L1→L2→L3，跑完释放再下一个。
"""
import glob, os, re, subprocess, sys, tempfile, time
import numpy as np

os.environ["HB_DNN_USER_DEFINED_L2M_SIZES"] = "6:6:6:6"
from pyCauchyKesai import CauchyKesai

HBM_DIR = "/root/ssd/OELLM_Runtime/robotrea_model/v1"
FRAMES = 10
DT = {"float16": np.float16, "float32": np.float32, "int32": np.int32, "int64": np.int64}

TESTS = [
    ("siglip",       "siglip",        "1,2,3,4", 1, ""),
    ("gemma_expert", "gemma_expert",  "1,2,3,4", 4, ""),
    ("gemma_llm",    "gemma",         "1,2,3,4", 4, ""),
    ("gemma_llm",    "gemma",         "1,2,3,4", 4, "hbdk20260318_EnableHpcTrue"),
    ("gemma_llm",    "gemma",         "1,2,3,4", 4, "hbdk20260331_EnableHpcTrue"),
    ("gemma_llm",    "gemma",         "1,2,3,4", 4, "hbdk20260318_EnableHpcFlase"),
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

def run_hrt_perf(hbm, mn, cores, input_files=None):
    """Run hrt_model_exec perf. Returns avg_ms or None."""
    cmd = ["hrt_model_exec", "perf", "--model_file", hbm, "--model_name", mn,
           "--core_id", cores, "--thread_num", "1", "--frame_count", str(FRAMES)]
    if input_files:
        cmd += ["--input_file", ",".join(input_files)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
        env={**os.environ, "HB_DNN_USER_DEFINED_L2M_SIZES": "6:6:6:6"})
    combined = proc.stderr + proc.stdout
    m = re.search(r"Average\s+latency\s+is:\s+([\d.]+)\s+ms", combined)
    fps = None
    m2 = re.search(r"Frame\s+rate\s+is:\s+([\d.]+)\s+FPS", combined)
    if m2: fps = float(m2.group(1))
    if m: return float(m.group(1)), fps
    return None, None


# =============================================================================
# main
# =============================================================================
print("=" * 80)
print("  pyCauchyKesai vs hbm_runtime vs hrt_model_exec perf — 三层对比")
print(f"  帧数: {FRAMES}, 单线程, 各自独立随机脏数据, 串行执行")
print("=" * 80)

results = []  # [(label, l1_wall, l1_bpu, l2_avg, l3_avg), ...]

for kw, mn, cores, ncores, pref in TESTS:
    try:
        hbm = find_hbm(kw, pref)
    except FileNotFoundError:
        print(f"\n  SKIP: {kw}/{pref} not found")
        continue
    lb = label(hbm)
    print(f"\n{'─'*60}\n  [{lb}] {os.path.basename(hbm)}\n{'─'*60}")

    l1_wall = l1_bpu = l2_avg = l3_avg = None
    pyck_inputs = None  # reuse same dirty data for L1 and L2

    # ---- L1: pyCauchyKesai ----
    try:
        print(f"  [L1-pyCK] loading ...", end=" ", flush=True)
        model = CauchyKesai(hbm, n_task=1, model_cnt_select=0)
        if ncores > 1:
            model.set_scheduling_params(bpu_cores=list(range(ncores)))
        info = model.s()
        # gen dirty inputs
        pyck_inputs = []
        for inp in info["inputs"]:
            shape = tuple(inp["shape"]); dtype = DT[inp["dtype"]]
            if inp["dtype"] in ("float16","float32"):
                data = np.random.randn(*shape).astype(dtype)
            else:
                ii = np.iinfo(dtype); data = np.random.randint(ii.min//2, ii.max//2, shape, dtype=dtype)
            pyck_inputs.append(np.ascontiguousarray(data))
        # warmup
        model.inference(pyck_inputs)
        # benchmark 10
        walls = []; bpus = []
        for _ in range(FRAMES):
            t0 = time.perf_counter()
            model.inference(pyck_inputs)
            t1 = time.perf_counter()
            walls.append((t1-t0)*1000)
            bpus.append(model.t()["time_ms"])
        l1_wall = float(np.mean(walls)); l1_bpu = float(np.mean(bpus))
        print(f"wall={l1_wall:.2f}ms  bpu={l1_bpu:.2f}ms  "
              f"(min={min(walls):.2f} max={max(walls):.2f})")
        # Save inputs as .bin for L2
        with tempfile.TemporaryDirectory() as td:
            bin_paths = []
            for i, arr in enumerate(pyck_inputs):
                p = f"{td}/in_{i}.bin"; arr.tofile(p); bin_paths.append(p)
            del model  # release before hrt_model_exec

            # ---- L2: hbm_runtime (perf + same dirty .bin) ----
            print(f"  [L2-hbmRT] ...", end=" ", flush=True)
            l2_avg, l2_fps = run_hrt_perf(hbm, mn, cores, bin_paths)
            if l2_avg:
                print(f"avg={l2_avg:.2f}ms  fps={l2_fps:.1f}")
            else:
                print("FAIL")
    except Exception as e:
        print(f"FAIL: {e}")
        try: del model
        except: pass

    # ---- L3: hrt_model_exec perf (auto data) ----
    print(f"  [L3-hrtP] ...", end=" ", flush=True)
    l3_avg, l3_fps = run_hrt_perf(hbm, mn, cores, None)
    if l3_avg:
        print(f"avg={l3_avg:.2f}ms  fps={l3_fps:.1f}")
    else:
        print("FAIL")

    results.append((lb, l1_wall, l1_bpu, l2_avg, l3_avg))
    time.sleep(0.3)

# ---- Table ----
print("\n"); print("=" * 95)
print("                        三 层 对 比 结 果")
print("=" * 95)
hdr = (f"{'模型':<28s} | {'L1 pyCK wall':>12s} | {'L2 hbmRT bin':>13s} | "
       f"{'L3 hrtP auto':>13s} | {'L1-L2':>8s} | {'L1-L3':>8s} | {'L2-L3':>8s}")
print(hdr); print("-" * 95)
for lb, l1w, l1b, l2a, l3a in results:
    if l1w is None or l2a is None or l3a is None:
        print(f"{lb:<28s} | {'FAIL':>12s} | {'FAIL':>13s} | {'FAIL':>13s}")
        continue
    print(f"{lb:<28s} | {l1w:10.2f}ms | {l2a:11.2f}ms | {l3a:11.2f}ms | "
          f"{l1w-l2a:+6.2f}ms | {l1w-l3a:+6.2f}ms | {l2a-l3a:+6.2f}ms")
print("-" * 95)
print("""
  L1 pyCK  = pyCauchyKesai.inference() perf_counter 端到端 wall time
  L2 hbmRT = hrt_model_exec perf --input_file (C++原生, 与L1相同脏 .bin)
  L3 hrtP  = hrt_model_exec perf 无输入 (C++原生, 工具自生成随机数据)

  L1-L2  = Python 层 overhead (调度+拷贝+BPU提交)
  L1-L3  = L1-L2 + 输入方式差异 (.bin vs 自生成)
  L2-L3  = 运行时相同, 仅输入不同 (.bin vs 自生成)
  pyCK bpu = CauchyKesai.t() BPU纯推理时间
""")

print("=" * 80)
print(" L1: pyCauchyKesai wall vs bpu")
print("=" * 80)
for lb, l1w, l1b, l2a, l3a in results:
    if l1w is None: continue
    print(f"  {lb:<28s} wall={l1w:.2f}ms  bpu={l1b:.2f}ms  diff={l1w-l1b:+.2f}ms")
