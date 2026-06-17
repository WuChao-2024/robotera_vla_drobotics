#!/usr/bin/env python3
"""
Pi05 Gemma LLM bc→HBM 编译工具。

从已有的 .convert.bc 文件编译为 HBM 模型，并运行 perf 生成 html/json 报告。
无需加载原始模型权重，无需校准数据。

用法:
  conda activate oellm_build_1.0.4_
  cd /home/chao.wu/pi_oellm/ROBOTREA/robotera_vla_drobotics/scripts_drobotics/oellm_cauchy
  python pi05_oellm_convert_bcGemmaLLM.py \
      --convert-bc-path /home/chao.wu/pi_oellm/ROBOTREA/M7_pickplace_example_ckpt_hbm1/gemma_llm_hbdk20260318_EnableHpcFlase/pi05_gemma_llm_pi05_action_horizon_20_ptq.convert.bc \
      --output-hbm-path /home/chao.wu/pi_oellm/ROBOTREA/M7_pickplace_example_ckpt_hbm1/gemma_llm_hbdk20260318_EnableHpcFlase/pi05_gemma_llm_pi05_action_horizon_20_ptq.hbm \
      --march nash-p \
      --jobs 48 \
      --enable-hpc False

  conda activate oellm_build_1.0.4_
  cd /home/chao.wu/pi_oellm/ROBOTREA/robotera_vla_drobotics/scripts_drobotics/oellm_cauchy
  python pi05_oellm_convert_bcGemmaLLM.py \
      --convert-bc-path /home/chao.wu/pi_oellm/ROBOTREA/M7_pickplace_example_ckpt_hbm1/gemma_llm_hbdk20260318_EnableHpcTrue/pi05_gemma_llm_pi05_action_horizon_20_ptq.convert.bc \
      --output-hbm-path /home/chao.wu/pi_oellm/ROBOTREA/M7_pickplace_example_ckpt_hbm1/gemma_llm_hbdk20260318_EnableHpcTrue/pi05_gemma_llm_pi05_action_horizon_20_ptq.hbm \
      --march nash-p \
      --jobs 48 \
      --enable-hpc True

  conda activate robotera_vla_drobotics_convert
  cd /home/chao.wu/pi_oellm/ROBOTREA/robotera_vla_drobotics/scripts_drobotics/oellm_cauchy
  python pi05_oellm_convert_bcGemmaLLM.py \
      --convert-bc-path /home/chao.wu/pi_oellm/ROBOTREA/M7_pickplace_example_ckpt_hbm1/gemma_llm_hbdk20260331_EnableHpcFlase/pi05_gemma_llm_pi05_action_horizon_20_ptq.convert.bc \
      --output-hbm-path /home/chao.wu/pi_oellm/ROBOTREA/M7_pickplace_example_ckpt_hbm1/gemma_llm_hbdk20260331_EnableHpcFlase/pi05_gemma_llm_pi05_action_horizon_20_ptq.hbm \
      --march nash-p \
      --jobs 48 \
      --enable-hpc False


  conda activate robotera_vla_drobotics_convert
  cd /home/chao.wu/pi_oellm/ROBOTREA/robotera_vla_drobotics/scripts_drobotics/oellm_cauchy
  python pi05_oellm_convert_bcGemmaLLM.py \
      --convert-bc-path /home/chao.wu/pi_oellm/ROBOTREA/M7_pickplace_example_ckpt_hbm1/gemma_llm_hbdk20260331_EnableHpcTrue/pi05_gemma_llm_pi05_action_horizon_20_ptq.convert.bc \
      --output-hbm-path /home/chao.wu/pi_oellm/ROBOTREA/M7_pickplace_example_ckpt_hbm1/gemma_llm_hbdk20260331_EnableHpcTrue/pi05_gemma_llm_pi05_action_horizon_20_ptq.hbm \
      --march nash-p \
      --jobs 48 \
      --enable-hpc True

  conda activate oellm_build_1.0.4_hbdk4.10.2a2.dev20260312
  cd /home/chao.wu/pi_oellm/ROBOTREA/robotera_vla_drobotics/scripts_drobotics/oellm_cauchy
  python pi05_oellm_convert_bcGemmaLLM.py \
      --convert-bc-path /home/chao.wu/pi_oellm/ROBOTREA/M7_pickplace_example_ckpt_hbm1/gemma_llm_hbdk20260312_EnableHpcFlase/pi05_gemma_llm_pi05_action_horizon_20_ptq.convert.bc \
      --output-hbm-path /home/chao.wu/pi_oellm/ROBOTREA/M7_pickplace_example_ckpt_hbm1/gemma_llm_hbdk20260312_EnableHpcFlase/pi05_gemma_llm_pi05_action_horizon_20_ptq.hbm \
      --march nash-p \
      --jobs 48 \
      --enable-hpc False


  conda activate oellm_build_1.0.4_hbdk4.10.2a2.dev20260312
  cd /home/chao.wu/pi_oellm/ROBOTREA/robotera_vla_drobotics/scripts_drobotics/oellm_cauchy
  python pi05_oellm_convert_bcGemmaLLM.py \
      --convert-bc-path /home/chao.wu/pi_oellm/ROBOTREA/M7_pickplace_example_ckpt_hbm1/gemma_llm_hbdk20260312_EnableHpcTrue/pi05_gemma_llm_pi05_action_horizon_20_ptq.convert.bc \
      --output-hbm-path /home/chao.wu/pi_oellm/ROBOTREA/M7_pickplace_example_ckpt_hbm1/gemma_llm_hbdk20260312_EnableHpcTrue/pi05_gemma_llm_pi05_action_horizon_20_ptq.hbm \
      --march nash-p \
      --jobs 48 \
      --enable-hpc True
"""

import argparse
import os
import sys
import time
from pathlib import Path

# 确保当前目录在 sys.path 中，使 leap_llm 可以被导入
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# OE-LLM 编译环境变量（与 pi05_oellm_convert.py 一致）
os.environ.setdefault("DEV_B30_TRITON_VPU", "1")
os.environ.setdefault("DEV_B30_ENABLE_VPU_EXTRA_OP", "1")
os.environ.setdefault("DEV_B30_ENABLE_VPU_TRIAL_OP", "1")

from hbdk4.compiler import compile, link, load
from hbdk4.compiler.hbm_tools import hbm_perf


def compile_gemma_llm_bc_to_hbm(
    convert_bc_path: str,
    output_hbm_path: str,
    compile_kwargs: dict,
):
    """从 .convert.bc 编译 Gemma LLM 到 HBM，并运行 perf。

    流程与 model_gemma.py 的 LanguageModel.compile() 中
    up_to="convert_bc" 之后的阶段完全一致：
      1. load(.convert.bc) → mlir_module
      2. compile(mlir_module, .hbo, **kwargs)  — kwargs 包含 core_num=4, max_l2m_size=25165824
      3. link([hbo], .hbm)
      4. hbm_perf(.hbm, output_dir) → html/json 报告
    """
    # --- Step 1: 从 .convert.bc 加载 mlir_module ---
    print(f"\nLoading mlir_module from {convert_bc_path} ...")
    t0 = time.time()
    mlir_module = load(convert_bc_path)
    print(f"  Done in {time.time() - t0:.1f}s")

    # --- Step 2: 编译到 HBO ---
    # Gemma LLM 特有参数（与 model_gemma.py LanguageModel.compile() 一致）
    kwargs = dict(compile_kwargs)
    kwargs["core_num"] = 4
    kwargs["max_l2m_size"] = 25165824

    hbo_path = str(Path(output_hbm_path).with_suffix(".hbo"))
    print(f"\nCompiling HBO → {hbo_path}")
    print(f"  core_num={kwargs['core_num']}, max_l2m_size={kwargs['max_l2m_size']}, "
          f"march={kwargs['march']}, jobs={kwargs['jobs']}, opt={kwargs['opt']}, "
          f"enable_vpu={kwargs.get('enable_vpu', 'N/A')}, "
          f"enable_hpc={kwargs.get('enable_hpc', 'N/A')}")
    t0 = time.time()
    hbo_model = compile(
        mlir_module,
        hbo_path,
        **kwargs,
    )
    print(f"  HBO compile done in {time.time() - t0:.1f}s")

    # --- Step 3: 链接为 HBM ---
    print(f"\nLinking HBM → {output_hbm_path}")
    t0 = time.time()
    hbm_model = link([hbo_model], output_hbm_path)
    print(f"  Link done in {time.time() - t0:.1f}s")

    # --- Step 4: Perf 分析 ---
    output_dir = os.path.dirname(output_hbm_path)
    print(f"\nRunning perf analysis ...")
    t0 = time.time()
    ret = hbm_perf(output_hbm_path, output_dir=output_dir)
    if ret == 0:
        print(f"  Perf done in {time.time() - t0:.1f}s → {output_dir}")
    else:
        print(f"  Perf returned non-zero: {ret}")

    print("\n" + "=" * 60)
    print("Gemma LLM bc→HBM compilation complete:")
    print(f"  HBM: {output_hbm_path}")
    print(f"  Perf: {output_dir}/")
    print("=" * 60)


def parse_args():
    p = argparse.ArgumentParser(
        description="Pi05 Gemma LLM bc→HBM 编译工具",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # 必需参数
    p.add_argument(
        "--convert-bc-path", type=str, required=True,
        help="Path to the .convert.bc file "
             "(e.g. pi05_gemma_llm_pi05_action_horizon_20_ptq.convert.bc).",
    )
    p.add_argument(
        "--output-hbm-path", type=str, required=True,
        help="Path for the output .hbm file "
             "(e.g. pi05_gemma_llm_pi05_action_horizon_20_ptq.hbm).",
    )

    # 编译参数（与 pi05_oellm_convert.py 中 llm_kwargs 保持一致）
    p.add_argument(
        "--march", type=str, default="nash-p",
        choices=["nash-e", "nash-m", "nash-p"],
        help="Target hardware architecture (default: nash-p).",
    )
    p.add_argument(
        "--jobs", type=int, default=32,
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
    p.add_argument(
        "--enable-hpc", type=str,
    )

    return p.parse_args()


def main():
    args = parse_args()

    # 校验路径
    if not os.path.isfile(args.convert_bc_path):
        print(f"Error: --convert-bc-path '{args.convert_bc_path}' does not exist.")
        sys.exit(1)

    if not args.convert_bc_path.endswith(".convert.bc"):
        print(f"Error: --convert-bc-path must end with '.convert.bc', got '{args.convert_bc_path}'")
        sys.exit(1)

    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_hbm_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 构建编译参数（与 pi05_oellm_convert.py 中 llm_kwargs 基础参数一致）
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
        "enable_vpu": True,
        "enable_hpc": args.enable_hpc.lower() in ("true", "1", "yes"),
    }

    print("=" * 60)
    print("Pi05 Gemma LLM bc→HBM Converter")
    print(f"  Convert BC:   {args.convert_bc_path}")
    print(f"  Output HBM:   {args.output_hbm_path}")
    print(f"  March:        {args.march}")
    print(f"  Jobs:         {args.jobs}")
    print(f"  Opt level:    {args.opt}")
    print(f"  enable_vpu:   True")
    print(f"  enable_hpc:   {compile_kwargs['enable_hpc']}")
    print("=" * 60)

    compile_gemma_llm_bc_to_hbm(
        convert_bc_path=args.convert_bc_path,
        output_hbm_path=args.output_hbm_path,
        compile_kwargs=compile_kwargs,
    )


if __name__ == "__main__":
    main()
