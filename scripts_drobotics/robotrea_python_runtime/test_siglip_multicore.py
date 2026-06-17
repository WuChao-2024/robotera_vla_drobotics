#!/usr/bin/env python3
"""
SigLIP 多核并行测试 — 找到正确的 BPU 核心分发方式。

测试方案：
  A. n_task=3 + 默认调度（对照组）
  B. n_task=3 + set_scheduling_params(bpu_cores=[0,1,2])
  C. n_task=3 + set_scheduling_params(bpu_cores=[])  (ANY)
  D. 3 个独立实例，各自绑定单核 [0], [1], [2]
  E. n_task=3 + model_cnt_select=0/1/2

每种方案跑 3 路 SigLIP，同时采集 BPU 4 个核的 ratio，
分析哪些核在工作、总耗时多少。
"""

import os
import subprocess
import time
import threading

import numpy as np

os.environ["HB_DNN_USER_DEFINED_L2M_SIZES"] = "6:6:6:6"

from pyCauchyKesai import CauchyKesai

# =============================================================================
# BPU 监控
# =============================================================================

BPU_CORES = [
    "/sys/devices/platform/soc/2a108000.bpu/ratio",   # core 0
    "/sys/devices/platform/soc/28108000.bpu/ratio",   # core 1
    "/sys/devices/platform/soc/2b108000.bpu/ratio",   # core 2
    "/sys/devices/platform/soc/29108000.bpu/ratio",   # core 3
]

def read_bpu_ratios():
    """读取 4 个 BPU 核心的占用率。"""
    ratios = []
    for path in BPU_CORES:
        try:
            with open(path) as f:
                ratios.append(int(f.read().strip()))
        except:
            ratios.append(-1)
    return ratios


class BPUMonitor:
    """后台线程高频采样 BPU ratio。"""

    def __init__(self, interval_ms=20):
        self.interval = interval_ms / 1000.0
        self.samples = []
        self._stop = threading.Event()

    def start(self):
        self.samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self):
        while not self._stop.is_set():
            r = read_bpu_ratios()
            self.samples.append((time.perf_counter(), r))
            time.sleep(self.interval)

    def summary(self):
        """返回每个核的: max_ratio, avg_when_active, active_samples/total_samples。"""
        if not self.samples:
            return "no samples"
        n = len(self.samples)
        per_core = [[] for _ in range(4)]
        for _, ratios in self.samples:
            for c in range(4):
                per_core[c].append(ratios[c])
        lines = []
        for c in range(4):
            vals = per_core[c]
            active = [v for v in vals if v > 0]
            mx = max(vals) if vals else 0
            avg = sum(active) / len(active) if active else 0
            lines.append(f"  core{c}: max={mx:3d}%  avg_active={avg:5.1f}%  "
                         f"active={len(active)}/{n}")
        return "\n".join(lines)


# =============================================================================
# 测试数据
# =============================================================================

SIGLIP_PATH = "/root/ssd/OELLM_Runtime/robotrea_model/v1/pi05_siglip_pi05_action_horizon_20_ptq.hbm"

def make_test_inputs():
    """生成 3 路测试数据。"""
    imgs = []
    pos = np.arange(0, 256, dtype=np.int64).reshape(1, 256)
    for _ in range(3):
        img = np.random.randn(1, 3, 224, 224).astype(np.float16)
        imgs.append([np.ascontiguousarray(img), pos])
    return imgs


# =============================================================================
# 测试函数
# =============================================================================

def warmup_siglip(model, inputs):
    """跑 2 次预热。"""
    for inp in inputs:
        model.inference(inp)


def run_test(label, infer_fn, inputs, n_runs=5):
    """跑 n_runs 轮，采集 BPU ratio。"""
    print(f"\n{'='*60}")
    print(f"  Test: {label}")
    print(f"{'='*60}")

    # baseline
    print(f"  BPU idle: {read_bpu_ratios()}")

    times = []
    for i in range(n_runs):
        mon = BPUMonitor(interval_ms=10)
        mon.start()
        t0 = time.perf_counter()

        # 执行 3 路 siglip
        infer_fn(inputs)

        elapsed = time.perf_counter() - t0
        mon.stop()
        times.append(elapsed)

        if i == n_runs - 1:  # 只打印最后一轮的详情
            print(f"  [{i+1}] {elapsed*1000:.1f}ms")
            print(mon.summary())

    avg = np.mean(times) * 1000
    mn = np.min(times) * 1000
    print(f"  => avg={avg:.1f}ms, min={mn:.1f}ms (over {n_runs} runs)")


# =============================================================================
# 各方案实现
# =============================================================================

def test_a_default():
    """A. n_task=3, 默认调度。"""
    model = CauchyKesai(SIGLIP_PATH, n_task=3, model_cnt_select=0)
    inputs = make_test_inputs()
    warmup_siglip(model, inputs)

    def infer(inputs):
        for i in range(3):
            model.safe_start(inputs[i], task_id=i)
        for i in range(3):
            model.wait(task_id=i)

    run_test("A: n_task=3, default scheduling", infer, inputs)


def test_b_cores_012():
    """B. n_task=3 + bpu_cores=[0,1,2]。"""
    model = CauchyKesai(SIGLIP_PATH, n_task=3, model_cnt_select=0)
    model.set_scheduling_params(bpu_cores=[0, 1, 2])
    inputs = make_test_inputs()
    warmup_siglip(model, inputs)

    def infer(inputs):
        for i in range(3):
            model.safe_start(inputs[i], task_id=i)
        for i in range(3):
            model.wait(task_id=i)

    run_test("B: n_task=3 + bpu_cores=[0,1,2]", infer, inputs)


def test_c_cores_any():
    """C. n_task=3 + bpu_cores=[] (ANY)。"""
    model = CauchyKesai(SIGLIP_PATH, n_task=3, model_cnt_select=0)
    model.set_scheduling_params(bpu_cores=[])  # ANY
    inputs = make_test_inputs()
    warmup_siglip(model, inputs)

    def infer(inputs):
        for i in range(3):
            model.safe_start(inputs[i], task_id=i)
        for i in range(3):
            model.wait(task_id=i)

    run_test("C: n_task=3 + bpu_cores=[] (ANY)", infer, inputs)


def test_d_3_instances():
    """D. 3 个独立实例，各自绑定单核。"""
    m0 = CauchyKesai(SIGLIP_PATH, n_task=1, model_cnt_select=0)
    m0.set_scheduling_params(bpu_cores=[0])
    m1 = CauchyKesai(SIGLIP_PATH, n_task=1, model_cnt_select=0)
    m1.set_scheduling_params(bpu_cores=[1])
    m2 = CauchyKesai(SIGLIP_PATH, n_task=1, model_cnt_select=0)
    m2.set_scheduling_params(bpu_cores=[2])

    inputs = make_test_inputs()
    for m, inp in zip([m0, m1, m2], inputs):
        m.inference(inp)  # warmup

    def infer(inputs):
        m0.safe_start(inputs[0], task_id=0)
        m1.safe_start(inputs[1], task_id=0)
        m2.safe_start(inputs[2], task_id=0)
        m0.wait(task_id=0)
        m1.wait(task_id=0)
        m2.wait(task_id=0)

    run_test("D: 3 instances, cores=[0],[1],[2]", infer, inputs)


def test_e_model_cnt_select():
    """E. n_task=3, 不同 model_cnt_select 值。"""
    for mcs in [0, 1, 2]:
        try:
            model = CauchyKesai(SIGLIP_PATH, n_task=1, model_cnt_select=mcs)
            model.set_scheduling_params(bpu_cores=[mcs])
            inputs = make_test_inputs()
            model.inference(inputs[0])  # warmup
            print(f"  model_cnt_select={mcs}: loaded OK, bpu_core_num={model.bpu_core_num}")
        except Exception as e:
            print(f"  model_cnt_select={mcs}: FAILED - {e}")


# =============================================================================
# main
# =============================================================================

if __name__ == "__main__":
    print("SigLIP multi-core dispatch test")
    print(f"BPU idle ratios: {read_bpu_ratios()}")
    print()

    # 先检查 model_cnt_select 可选值
    test_e_model_cnt_select()

    # 逐方案测试
    test_a_default()
    test_b_cores_012()
    test_c_cores_any()
    test_d_3_instances()

    print("\n" + "=" * 60)
    print("Done.")
