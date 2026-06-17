#!/usr/bin/env python3
"""
Pi05 模型 OE-LLM 转换工具（独立版）。

从 safetensors 权重编译为 3 个 HBM 子模型，输出目录结构：

  output_path/
  ├── siglip/
  │   ├── *.hbm, *.bc, *.convert.bc
  │   └── perf *.html, *.json
  ├── gemma_llm/
  │   ├── *.hbm, *.bc, *.convert.bc
  │   └── perf *.html, *.json
  └── gemma_expert/
      ├── *.hbm, *.bc, *.convert.bc
      └── perf *.html, *.json

action_dim / action_horizon 从模型目录的 config.json 读取。
校准数据使用 LeRobot Dataset v2.1 格式（与 compare.py 一致）。

用法:

  conda activate oellm_build_1.0.4_hbdk4.10.2a2.dev20260318
  cd /home/chao.wu/pi_oellm/ROBOTREA/robotera_vla_drobotics/scripts_drobotics/oellm_cauchy/
  python /home/chao.wu/pi_oellm/ROBOTREA/robotera_vla_drobotics/scripts_drobotics/oellm_cauchy/pi05_oellm_convert.py \
    --model-path /home/chao.wu/pi_oellm/ROBOTREA/M7_pickplace_example_ckpt_pt \
    --output-path /home/chao.wu/pi_oellm/ROBOTREA/M7_pickplace_example_ckpt_hbm_v2_hbdk20260318 \
    --dataset-root /home/chao.wu/pi_oellm/ROBOTREA/M7_pickplace_example_merged_LeRobotDataSetv2.1 \
    --cal-num 10000 \
    --seed 42 \
    --device cuda:0

  conda activate robotera_vla_drobotics_convert
  cd /home/chao.wu/pi_oellm/ROBOTREA/robotera_vla_drobotics/scripts_drobotics/oellm_cauchy/
  python /home/chao.wu/pi_oellm/ROBOTREA/robotera_vla_drobotics/scripts_drobotics/oellm_cauchy/pi05_oellm_convert.py \
    --model-path /home/chao.wu/pi_oellm/ROBOTREA/M7_pickplace_example_ckpt_pt \
    --output-path /home/chao.wu/pi_oellm/ROBOTREA/M7_pickplace_example_ckpt_hbm_v2_hbdk20260331 \
    --dataset-root /home/chao.wu/pi_oellm/ROBOTREA/M7_pickplace_example_merged_LeRobotDataSetv2.1 \
    --cal-num 10000 \
    --seed 42 \
    --device cuda:1
    
  conda activate oellm_build_1.0.4_hbdk4.10.2a2.dev20260312
  cd /home/chao.wu/pi_oellm/ROBOTREA/robotera_vla_drobotics/scripts_drobotics/oellm_cauchy/
  python /home/chao.wu/pi_oellm/ROBOTREA/robotera_vla_drobotics/scripts_drobotics/oellm_cauchy/pi05_oellm_convert.py \
    --model-path /home/chao.wu/pi_oellm/ROBOTREA/M7_pickplace_example_ckpt_pt \
    --output-path /home/chao.wu/pi_oellm/ROBOTREA/M7_pickplace_example_ckpt_hbm_hbdk20260312 \
    --dataset-root /home/chao.wu/pi_oellm/ROBOTREA/M7_pickplace_example_merged_LeRobotDataSetv2.1 \
    --cal-num 3000 \
    --seed 42 \
    --device cuda:1
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# 确保当前目录在 sys.path 中，使 leap_llm 可以被导入
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# OE-LLM 编译环境变量
os.environ.setdefault("DEV_B30_TRITON_VPU", "1")
os.environ.setdefault("DEV_B30_ENABLE_VPU_EXTRA_OP", "1")
os.environ.setdefault("DEV_B30_ENABLE_VPU_TRIAL_OP", "1")


import numpy as np
import torch


from pi05_leap_model import Pi05LeapModel
from lerobot_v21_reader import load_calib_samples


# =============================================================================
# Pi05 模型编译
# =============================================================================

class Pi05Converter:
    """Pi05 模型加载、校准、编译。"""

    def __init__(
        self,
        model_path: str,
        output_path: str,
        dataset_root: str,
        cal_num: int = 100,
        seed: int = 42,
        device: str = "cuda:0",
        model_type: str = "pi05",
    ):
        self.model_path = model_path
        self.output_path = output_path
        self.device = device
        self.model_type = model_type

        # --- 从 config.json 读取模型参数 ---
        config_file = os.path.join(model_path, "config.json")
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"config.json not found in {model_path}")
        with open(config_file) as f:
            model_meta = json.load(f)
        self.action_dim = model_meta["action_dim"]
        self.action_horizon = model_meta["action_horizon"]

        # --- 加载纯模型（三子模型 + tokenizer + 配置推断） ---
        safetensors_file = os.path.join(model_path, "model.safetensors")
        if not os.path.isfile(safetensors_file):
            raise FileNotFoundError(f"model.safetensors not found in {model_path}")

        print(f"Loading Pi05LeapModel from {model_path} (action_horizon={self.action_horizon})...")
        self._leap_model = Pi05LeapModel(model_path, action_horizon=self.action_horizon, device=device, use_softmax_mask=True)

        self.model_config = self._leap_model.model_config
        self.vision_token_num = self._leap_model.vision_token_num
        self.model_siglip = self._leap_model.siglip
        self.model_gemma_llm = self._leap_model.gemma_llm
        self.model_gemma_expert = self._leap_model.gemma_expert

        print(f"  action_dim={self.action_dim}, action_horizon={self.action_horizon}")
        print(f"  vision_token_num={self.vision_token_num} (num_patches, no pruning)")
        print(f"  lm_hidden_size={self.model_config['lm_hidden_size']}, "
              f"expert_hidden_size={self.model_config['expert_hidden_size']}")

        # 子目录：每个模型独立文件夹
        hbm_name = f"{model_type}_action_horizon_{self.action_horizon}_ptq"
        self.siglip_dir = os.path.join(output_path, "siglip")
        self.gemma_llm_dir = os.path.join(output_path, "gemma_llm")
        self.gemma_expert_dir = os.path.join(output_path, "gemma_expert")
        for d in [self.siglip_dir, self.gemma_llm_dir, self.gemma_expert_dir]:
            os.makedirs(d, exist_ok=True)

        self.siglip_hbm = os.path.join(self.siglip_dir, f"{model_type}_siglip_{hbm_name}.hbm")
        self.gemma_llm_hbm = os.path.join(self.gemma_llm_dir, f"{model_type}_gemma_llm_{hbm_name}.hbm")
        self.gemma_expert_hbm = os.path.join(self.gemma_expert_dir, f"{model_type}_gemma_expert_{hbm_name}.hbm")

        # 加载校准数据（LeRobot Dataset v2.1，与 compare.py 一致）
        print(f"\nLoading calibration data from {dataset_root} ...")
        self.calib_samples = load_calib_samples(dataset_root, cal_num, seed)
        print(f"  Loaded {len(self.calib_samples)} calibration samples")

        print("Model weights loaded successfully.")

    def _calibrate_forward(self, dtype: torch.dtype, action_horizon: int):
        """运行校准前向，收集量化统计信息。

        数据管线与 Server infer() 完全一致：
          preprocess_image_numpy → tokenize → forward(seed=42)
        """
        device = self.device
        start_time = time.time()

        for calib_index, sample in enumerate(self.calib_samples):
            prompt = sample["prompt"]

            # 图像预处理（与 Server infer() 一致：cv2.INTER_AREA）
            images = []
            for cam_key in ["cam_high", "cam_left_wrist", "cam_right_wrist"]:
                img = self._leap_model.preprocess_image_numpy(sample["images"][cam_key])
                images.append(img.to(device).to(dtype=dtype))

            # Tokenize
            tokens, valid_token_len = self._leap_model.tokenize(prompt)

            # 模型前向（seed=42 与 Server infer() 一致）
            self._leap_model.forward(images, tokens, valid_token_len,
                                      action_horizon=action_horizon, seed=42)

            if (calib_index + 1) % 5 == 0:
                print(f"  Calibration progress: {calib_index + 1}/{len(self.calib_samples)}")

        elapsed = time.time() - start_time
        print(f"Calibration completed in {elapsed:.1f}s for {len(self.calib_samples)} samples.")

    @staticmethod
    def _run_perf(hbm_path: str, output_dir: str):
        """对 HBM 模型运行本地性能分析，生成 html/json 报告。"""
        from hbdk4.compiler.hbm_tools import hbm_perf
        print(f"  Running perf analysis...")
        t0 = time.time()
        ret = hbm_perf(hbm_path, output_dir=output_dir)
        if ret == 0:
            print(f"  Perf done in {time.time() - t0:.1f}s → {output_dir}")
        else:
            print(f"  Perf returned non-zero: {ret}")

    def compile(self, compile_kwargs: dict):
        """编译三个子模型为 HBM，并运行 perf。

        流程分三阶段：
          Phase 1: Calibration forward — 收集量化统计
          Phase 2: Export bc + convert.bc — 生成所有子模型的 bitcode
          Phase 3: Compile HBM + Perf — 从 bc/convert.bc 编译到 HBM 并 perf
        """
        device = self.device if torch.cuda.is_available() and self.device.startswith("cuda") else "cpu"
        calib_dtype = torch.float16   # 与官方 SDK 一致
        compile_dtype = torch.float16

        # --- Phase 1: 校准前向 ---
        print("\n" + "=" * 60)
        print("Phase 1: Calibration forward pass")
        print("=" * 60)
        self.model_siglip.model.to(device=device, dtype=calib_dtype)
        self.model_gemma_llm.model.to(device=device, dtype=calib_dtype)
        self.model_gemma_expert.model.to(device=device, dtype=calib_dtype)
        self.model_siglip.model.compile_mode(False)
        self.model_gemma_llm.model.compile_mode(False)
        self.model_gemma_expert.model.compile_mode(False)

        self._calibrate_forward(dtype=calib_dtype, action_horizon=self.action_horizon)

        # --- Phase 2: 生成所有 bc + convert.bc ---
        print("\n" + "=" * 60)
        print("Phase 2: Export bc + convert.bc for all sub-models")
        print("=" * 60)
        self.model_siglip.model.compile_mode(True)
        self.model_gemma_llm.model.compile_mode(True)
        self.model_gemma_expert.model.compile_mode(True)
        self.model_siglip.model.to(device="cpu", dtype=compile_dtype)
        self.model_gemma_llm.model.to(device="cpu", dtype=compile_dtype)
        self.model_gemma_expert.model.to(device="cpu", dtype=compile_dtype)

        siglip_kwargs = dict(compile_kwargs)
        siglip_kwargs["enable_vpu"] = True
        siglip_kwargs["enable_spu"] = False

        llm_kwargs = dict(compile_kwargs)
        llm_kwargs["enable_vpu"] = True
        llm_kwargs["enable_hpc"] = True

        expert_kwargs = dict(compile_kwargs)
        expert_kwargs["enable_vpu"] = True
        expert_kwargs["enable_hpc"] = True

        # Siglip
        print(f"\nExporting Siglip bc + convert.bc → {self.siglip_hbm}")
        t0 = time.time()
        self.model_siglip.compile(
            output_model_path=self.siglip_hbm,
            up_to="convert_bc",
            **siglip_kwargs,
        )
        print(f"  Done in {time.time() - t0:.1f}s")

        # Gemma LLM
        print(f"\nExporting Gemma LLM bc + convert.bc → {self.gemma_llm_hbm}")
        t0 = time.time()
        self.model_gemma_llm.compile(
            output_model_path=self.gemma_llm_hbm,
            up_to="convert_bc",
            **llm_kwargs,
        )
        print(f"  Done in {time.time() - t0:.1f}s")

        # Gemma Expert
        print(f"\nExporting Gemma Expert bc + convert.bc → {self.gemma_expert_hbm}")
        t0 = time.time()
        self.model_gemma_expert.compile(
            output_model_path=self.gemma_expert_hbm,
            up_to="convert_bc",
            **expert_kwargs,
        )
        print(f"  Done in {time.time() - t0:.1f}s")

        # --- Phase 3: 从 bc/convert.bc 编译到 HBM + perf ---
        print("\n" + "=" * 60)
        print("Phase 3: Compile HBM + Perf")
        print("=" * 60)

        # Siglip
        print(f"\nCompiling Siglip HBM → {self.siglip_hbm}")
        t0 = time.time()
        self.model_siglip.compile(
            output_model_path=self.siglip_hbm,
            **siglip_kwargs,
        )
        print(f"  Done in {time.time() - t0:.1f}s")
        self._run_perf(self.siglip_hbm, self.siglip_dir)

        # Gemma LLM
        print(f"\nCompiling Gemma LLM HBM → {self.gemma_llm_hbm}")
        t0 = time.time()
        self.model_gemma_llm.compile(
            output_model_path=self.gemma_llm_hbm,
            **llm_kwargs,
        )
        print(f"  Done in {time.time() - t0:.1f}s")
        self._run_perf(self.gemma_llm_hbm, self.gemma_llm_dir)

        # Gemma Expert
        print(f"\nCompiling Gemma Expert HBM → {self.gemma_expert_hbm}")
        t0 = time.time()
        self.model_gemma_expert.compile(
            output_model_path=self.gemma_expert_hbm,
            **expert_kwargs,
        )
        print(f"  Done in {time.time() - t0:.1f}s")
        self._run_perf(self.gemma_expert_hbm, self.gemma_expert_dir)

        print("\n" + "=" * 60)
        print("Compilation complete:")
        print(f"  Siglip:       {self.siglip_dir}/")
        print(f"  Gemma LLM:    {self.gemma_llm_dir}/")
        print(f"  Gemma Expert: {self.gemma_expert_dir}/")
        print("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Pi05 模型 OE-LLM 转换工具",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # 必需参数
    p.add_argument(
        "--model-path", type=str, required=True,
        help="Path to the safetensors model directory "
             "(containing model.safetensors and paligemma_tokenizer.model).",
    )
    p.add_argument(
        "--output-path", type=str, required=True,
        help="Path to save the compiled HBM models.",
    )
    p.add_argument(
        "--dataset-root", type=str, required=True,
        help="Path to LeRobot v2.1 dataset root directory "
             "(same format as compare.py).",
    )

    # 校准参数
    p.add_argument(
        "--cal-num", type=int, default=100,
        help="Number of calibration samples (default: 100).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling calibration data (default: 42).",
    )

    # 模型参数
    p.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device for calibration: 'cpu', 'cuda', 'cuda:0', etc. (default: cuda:0).",
    )

    # 编译参数
    p.add_argument(
        "--march", type=str, default="nash-p",
        choices=["nash-e", "nash-m", "nash-p"],
        help="Target hardware architecture (default: nash-p).",
    )
    p.add_argument(
        "--jobs", type=int, default=64,
        help="Number of parallel jobs during compilation (default: 32).",
    )
    p.add_argument(
        "--opt", type=int, default=2,
        choices=[0, 1, 2],
        help="Optimization level (default: 2).",
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Enable debug mode during compilation.",
    )
    p.add_argument(
        "--max-time-per-fc", type=float, default=0.0,
        help="Maximum time constraint (us) per func call (default: 0.0 = unlimited).",
    )
    p.add_argument(
        "--advice", type=float, default=0.0,
        help="HBO advice value (default: 0.0).",
    )
    p.add_argument(
        "--balance", type=int, default=100,
        help="Balance factor (default: 100).",
    )

    return p.parse_args()


def main():
    args = parse_args()

    # 校验路径
    if not os.path.exists(args.model_path):
        print(f"Error: --model-path '{args.model_path}' does not exist.")
        sys.exit(1)

    if not os.path.exists(args.dataset_root):
        print(f"Error: --dataset-root '{args.dataset_root}' does not exist.")
        sys.exit(1)

    tokenizer_file = os.path.join(args.model_path, "paligemma_tokenizer.model")
    if not os.path.isfile(tokenizer_file):
        print(f"Error: paligemma_tokenizer.model not found in --model-path '{args.model_path}'")
        sys.exit(1)

    # 构建编译参数
    compile_kwargs = {
        "march": args.march,
        "jobs": args.jobs,
        "progress_bar": True,
        "max_time_per_fc": args.max_time_per_fc,
        "opt": args.opt,
        "debug": args.debug,
        "advice": args.advice,
        "balance": args.balance,
        "input_no_padding": True,
        "output_no_padding": True,
    }

    print("=" * 60)
    print("Pi05 OE-LLM Converter")
    print(f"  Model:         {args.model_path}")
    print(f"  Output:        {args.output_path}")
    print(f"  Dataset:       {args.dataset_root}")
    print(f"  Cal samples:   {args.cal_num} (seed={args.seed})")
    print(f"  Device:        {args.device}")
    print(f"  March:         {args.march}")
    print(f"  Jobs:          {args.jobs}")
    print(f"  Opt level:     {args.opt}")
    print("=" * 60)

    converter = Pi05Converter(
        model_path=args.model_path,
        output_path=args.output_path,
        dataset_root=args.dataset_root,
        cal_num=args.cal_num,
        seed=args.seed,
        device=args.device,
    )

    converter.compile(compile_kwargs)


if __name__ == "__main__":
    main()
