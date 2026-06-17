#!/usr/bin/env python3
"""
BPU Pi05 VLA 推理封装 — ION 零拷贝 + 可选 Butterworth 低通滤波（最终版）。

核心优化：
  - KV Cache ION 零拷贝（消除 float16→float32→float16 + 36 次 memcpy）
  - SigLIP 3 实例多核并行
  - CPU/BPU 流水线重叠（Expert mask/pos 与 LLM BPU 并行执行）
  - 可选 Butterworth 低通滤波（filter_cutoff=None 则不启用）

数据流：
  CPU → cached ION (tokens, masks, vision) → LLM 输入
  LLM → uncached ION (36 KV) → Expert 输入（零拷贝！）
  Expert → uncached ION (actions) → CPU 读取

用法:
  from bpu_robotrea_pi05model_ion import BPU_ROBOTREA_VLA_Pi05_ION

  model = BPU_ROBOTREA_VLA_Pi05_ION(
      siglip_path=".../siglip.hbm",
      gemma_llm_path=".../gemma_llm.hbm",
      expert_path=".../expert.hbm",
      share_dir=".../share",
      urdf_path=".../l3_4.urdf",
      filter_cutoff=1.0,       # None = 不启用滤波
      filter_fs=50.0,
      filter_channels=16,
      filter_order=2,
      filter_steps=5,          # 只滤波前5步（客户端实际使用的步数）
  )
  actions = model.inference(images_dict, state_array, "pick the apple")
"""

import json
import logging
import os

import numpy as np

# L2 Cache 必须在 import pyCauchyKesai 之前设置
os.environ["HB_DNN_USER_DEFINED_L2M_SIZES"] = "6:6:6:6"

from pyCauchyKesai import IONArray, CauchyKesai

from bpu_robotrea_pi05model import (
    VISION_PATCH, NUM_VIEWS, VISION_PREFIX_LEN, NUM_LAYERS,
    ACTION_HORIZON, ACTION_DIM, STATE_DIM, CAM_KEYS,
    load_tokenizer, load_norm_stats, tokenize,
    build_gemma_attn_mask, make_gemma_position_ids, make_softmax_mask,
    build_expert_attn_mask, make_expert_position_ids,
    preprocess_image, unnormalize_actions,
    CamAbsoluteEeActions,
)

logger = logging.getLogger("bpu_robotrea_pi05_ion")

# =============================================================================
# BPU 对齐常量与工具
# =============================================================================

_BPU_ALIGN = 64  # S600 平台 64 字节对齐


def _aligned_nbytes(dtype_str: str, shape) -> int:
    """计算 BPU stride 对齐后的字节数。

    从最内层维度往外逐层计算 stride，每层向上取整到 64 字节对齐。
    这与 BPU 编译器内部的 fixup_aligned_stride 逻辑一致。
    """
    dtype = np.dtype(dtype_str)
    ndim = len(shape)
    if ndim == 0:
        return dtype.itemsize

    strides = [0] * ndim
    strides[-1] = dtype.itemsize
    for d in range(ndim - 2, -1, -1):
        raw = strides[d + 1] * shape[d + 1]
        strides[d] = (raw + _BPU_ALIGN - 1) & ~(_BPU_ALIGN - 1)

    return strides[0] * shape[0]


def _make_ion(dtype_str: str, shape, cached: bool = True) -> IONArray:
    """创建 IONArray，自动按 BPU stride 对齐分配足够字节。"""
    bs = _aligned_nbytes(dtype_str, tuple(shape))
    return IONArray(np.dtype(dtype_str), tuple(shape), byte_size=bs, cached=cached)


# =============================================================================
# EMA 帧间平滑滤波器
# =============================================================================

class FrameEMA:
    """帧间 EMA 平滑滤波器。

    核心思路：每帧推理输出一个 chunk（20步），客户端只取前 N 步执行。
    帧间衔接问题：新 chunk 的 step0 与上一帧末尾值不同 → 跳变 → 抖动。

    解决方案：记住上一帧最后输出值 prev_output，新帧用 EMA 从旧值过渡到新值：
      output[0] = alpha * actions[0] + (1 - alpha) * prev_output
      output[1] = alpha * actions[1] + (1 - alpha) * output[0]
      ...
    1阶、无过冲、无振铃，帧间自然衔接。

    alpha 控制新旧比例：
      alpha=1.0 → 不滤波，纯用新值
      alpha=0.3 → 30%新值 + 70%旧值，强平滑，过渡慢
      alpha=0.5 → 50%新值 + 50%旧值，适中平滑
    """

    def __init__(self, alpha: float, channels: int):
        self.alpha = alpha
        self.channels = channels
        self.prev_output = None  # 首帧前为 None

    def smooth(self, actions_chunk: np.ndarray, n_steps: int) -> np.ndarray:
        """对 chunk 前 n_steps 步做 EMA 帧间平滑。

        Args:
            actions_chunk: (horizon, channels) float32
            n_steps: 滤波前 N 步（客户端实际执行的步数）

        Returns:
            修改后的 actions_chunk（原地修改前 n_steps 行）
        """
        if self.prev_output is None:
            # 首帧：初始化 prev_output 为 step0 原值，不滤波
            self.prev_output = actions_chunk[0, :self.channels].copy().astype(np.float64)
            return actions_chunk

        alpha = self.alpha
        prev = self.prev_output  # 上一帧最后输出值

        for t in range(n_steps):
            new_val = actions_chunk[t, :self.channels].astype(np.float64)
            blended = alpha * new_val + (1.0 - alpha) * prev
            actions_chunk[t, :self.channels] = blended.astype(np.float32)
            prev = blended

        # 保存本帧最后输出，供下一帧衔接
        self.prev_output = prev.copy()
        return actions_chunk


# =============================================================================
# BPU_ROBOTREA_VLA_Pi05_ION — 最终版
# =============================================================================

class BPU_ROBOTREA_VLA_Pi05_ION:
    """Pi05 VLA 推理封装 — ION 零拷贝 + 可选 Butterworth 低通滤波。

    优化：
      - Gemma LLM → Expert KV Cache 零拷贝（共享 IONArray）
      - SigLIP 3 实例各绑单核，真正多核并行
      - Expert mask/pos 与 LLM BPU 执行重叠
      - 可选 Butterworth 低通滤波（filter_cutoff=None 则不启用）
    """

    def __init__(self,
                 siglip_path: str,
                 gemma_llm_path: str,
                 expert_path: str,
                 share_dir: str,
                 urdf_path: str = None,
                 filter_alpha: float = None,
                 filter_channels: int = 38,
                 filter_steps: int = 10):
        """
        Args:
            siglip_path: SigLIP .hbm 模型文件路径
            gemma_llm_path: Gemma LLM .hbm 模型文件路径
            expert_path: Gemma Expert .hbm 模型文件路径
            share_dir: 共享资源目录（tokenizer, norm_stats, config.json）
            urdf_path: URDF 路径（可选，用于 CamAbsoluteEeActions）
            filter_alpha: EMA 平滑系数 (0-1)，0或None不启用。0.5=适中，0.3=强平滑
            filter_channels: 滤波通道数（action 前 N 维）
            filter_steps: 只滤波前 N 步（None = 全部）
        """
        import time as _time

        # ---- 1. 加载共享资源 ----
        self._tokenizer = load_tokenizer(share_dir)
        self._norm_stats = load_norm_stats(share_dir)

        config_file = os.path.join(share_dir, "config.json")
        with open(config_file) as f:
            config = json.load(f)
        self.action_dim = config.get("action_dim", ACTION_DIM)
        self.action_horizon = config.get("action_horizon", ACTION_HORIZON)

        # ---- 2. Gemma Expert: _no_alloc 模式（先加载 Expert 避免 IOVA 冲突）----
        logger.info(f"Loading Gemma Expert HBM: {expert_path}")
        t0 = _time.time()
        self._expert = CauchyKesai(
            expert_path, n_task=1, model_cnt_select=0, _no_alloc=True)
        self._expert.set_scheduling_params(bpu_cores=[0, 1, 2, 3])
        logger.info(f"  Gemma Expert loaded in {_time.time() - t0:.1f}s (_no_alloc)")

        # ---- 3. Gemma LLM: _no_alloc 模式 ----
        logger.info(f"Loading Gemma LLM HBM: {gemma_llm_path}")
        t0 = _time.time()
        self._gemma_llm = CauchyKesai(
            gemma_llm_path, n_task=1, model_cnt_select=0, _no_alloc=True)
        self._gemma_llm.set_scheduling_params(bpu_cores=[0, 1, 2, 3])
        logger.info(f"  Gemma LLM loaded in {_time.time() - t0:.1f}s (_no_alloc)")

        # ---- 4. SigLIP: 3 个独立实例各绑单核 (方案 D, 真正多核并行) ----
        logger.info(f"Loading SigLIP HBM: {siglip_path}")
        t0 = _time.time()
        self._siglips = []
        for core_id in [0, 1, 2]:
            s = CauchyKesai(siglip_path, n_task=1, model_cnt_select=0)
            s.set_scheduling_params(bpu_cores=[core_id])
            self._siglips.append(s)
        self._siglip_pos = np.arange(0, VISION_PATCH, dtype=np.int64).reshape(1, VISION_PATCH)
        logger.info(f"  SigLIP loaded in {_time.time() - t0:.1f}s "
                    f"(3 instances, cores=[0],[1],[2])")

        # ---- 5. 解析模型元数据 → 推断形状参数 ----
        self._parse_model_info()

        # ---- 6. 分配 ION 内存 ----
        self._allocate_ions()

        # ---- 7. 绑定 ION 到模型 ----
        self._bind_models()

        # ---- 8. 初始化不变数据 ----
        self._init_constants()

        # ---- 9. URDF 后处理 ----
        if urdf_path and os.path.exists(urdf_path):
            self._cam_abs_transform = CamAbsoluteEeActions(urdf_path)
            logger.info(f"URDF transform enabled: {urdf_path}")
        else:
            self._cam_abs_transform = None
            logger.info("URDF transform disabled")

        # ---- 10. EMA 帧间平滑滤波（可选） ----
        self._filter_alpha = filter_alpha
        self._filter_channels = filter_channels
        self._filter_steps = filter_steps

        if filter_alpha and filter_alpha > 0:
            self._filter = FrameEMA(
                alpha=filter_alpha, channels=filter_channels,
            )
            logger.info(f"EMA filter enabled: alpha={filter_alpha}, "
                        f"channels={filter_channels}, "
                        f"steps={filter_steps or 'all'}")
        else:
            self._filter = None
            logger.info("EMA filter disabled")

        logger.info(f"BPU_ROBOTREA_VLA_Pi05_ION ready: "
                    f"action_dim={self.action_dim}, "
                    f"action_horizon={self.action_horizon}, "
                    f"lang_tokens={self._lang_token_count}, "
                    f"total_seq={self._total_seq_len}, "
                    f"kv_count={self._expert_kv_count}, "
                    f"filter={'ON' if self._filter else 'OFF'}")

    # -----------------------------------------------------------------
    # 模型元数据解析
    # -----------------------------------------------------------------

    def _parse_model_info(self):
        """从 HBM 模型元数据推断所有张量形状。"""
        gemma_info = self._gemma_llm.s()
        expert_info = self._expert.s()

        self._gemma_inputs = gemma_info["inputs"]
        self._gemma_outputs = gemma_info["outputs"]
        self._expert_inputs = expert_info["inputs"]
        self._expert_outputs = expert_info["outputs"]

        # LLM 参数
        self._lang_token_count = self._gemma_inputs[0]["shape"][1]
        self._total_seq_len = VISION_PREFIX_LEN + self._lang_token_count

        # Expert KV cache 数量 = 总输入数 - 4 (state, x_t, mask, pos_ids)
        self._expert_kv_count = len(self._expert_inputs) - 4
        if self._expert_kv_count != NUM_LAYERS * 2:
            logger.warning(
                f"Expert KV count={self._expert_kv_count}, "
                f"expected {NUM_LAYERS * 2}; "
                f"LLM outputs {NUM_LAYERS * 2} KV tensors. "
                f"Will bind first {self._expert_kv_count} KV to Expert.")

        # Expert KV seq_len
        self._expert_kv_seq_len = self._expert_inputs[4]["shape"][1]

        # 提取形状元组
        self._gemma_in_shapes = [tuple(inp["shape"]) for inp in self._gemma_inputs]
        self._gemma_out_shapes = [tuple(out["shape"]) for out in self._gemma_outputs]
        self._expert_in_shapes = [tuple(inp["shape"]) for inp in self._expert_inputs]
        self._expert_out_shapes = [tuple(out["shape"]) for out in self._expert_outputs]

        # 提取 dtype 字符串
        self._gemma_in_dtypes = [inp["dtype"] for inp in self._gemma_inputs]
        self._gemma_out_dtypes = [out["dtype"] for out in self._gemma_outputs]
        self._expert_in_dtypes = [inp["dtype"] for inp in self._expert_inputs]
        self._expert_out_dtypes = [out["dtype"] for out in self._expert_outputs]

        logger.info(f"  GEMMA LLM: {len(self._gemma_inputs)} inputs, "
                    f"{len(self._gemma_outputs)} outputs")
        logger.info(f"  EXPERT:    {len(self._expert_inputs)} inputs, "
                    f"{len(self._expert_outputs)} outputs")
        logger.info(f"  KV shape:  {self._gemma_out_shapes[1]}, "
                    f"dtype={self._gemma_out_dtypes[1]}")
        logger.info(f"  lang_token_count={self._lang_token_count}, "
                    f"total_seq_len={self._total_seq_len}, "
                    f"expert_kv_count={self._expert_kv_count}, "
                    f"expert_kv_seq_len={self._expert_kv_seq_len}")

    # -----------------------------------------------------------------
    # ION 内存分配
    # -----------------------------------------------------------------

    def _allocate_ions(self):
        """分配所有 ION 内存。"""
        total_ion_bytes = 0

        def _alloc(dtype_str, shape, cached=True, label=""):
            nonlocal total_ion_bytes
            ion = _make_ion(dtype_str, shape, cached=cached)
            total_ion_bytes += ion.mem_size
            if label:
                logger.info(f"  {label}: dtype={dtype_str}, shape={shape}, "
                            f"cached={cached}, mem={ion.mem_size / 1024:.0f}KB")
            return ion

        # ── Gemma LLM 输入 (5 个, 全部 cached: CPU 写 → BPU 读) ──
        self._ion_gemma_tokens = _alloc(
            'int32', self._gemma_in_shapes[0], label="LLM input[0] tokens")
        self._ion_gemma_vision = _alloc(
            'float16', self._gemma_in_shapes[1], label="LLM input[1] vision")
        self._ion_gemma_mask = _alloc(
            'float16', self._gemma_in_shapes[2], label="LLM input[2] mask")
        self._ion_gemma_pos = _alloc(
            'int32', self._gemma_in_shapes[3], label="LLM input[3] pos")
        self._ion_gemma_softmax = _alloc(
            'float16', self._gemma_in_shapes[4], label="LLM input[4] softmax")

        # ── 共享 KV Cache (36 个, uncached: 纯 BPU→BPU) ──
        kv_shape = self._gemma_out_shapes[1]  # (1, seq, 256)
        self._ion_kv = []
        for i in range(NUM_LAYERS * 2):
            ion = _make_ion('float16', kv_shape, cached=False)
            self._ion_kv.append(ion)
            total_ion_bytes += ion.mem_size
        kv_total_mb = sum(ion.mem_size for ion in self._ion_kv) / 1024 / 1024
        logger.info(f"  KV cache: {NUM_LAYERS * 2} IONs × {kv_shape}, "
                    f"uncached, total={kv_total_mb:.1f}MB")

        # ── Gemma LLM 输出: hidden (Expert 不用，但 set_outputs 需要) ──
        self._ion_gemma_hidden = _alloc(
            'float16', self._gemma_out_shapes[0], cached=False,
            label="LLM output[0] hidden")

        # ── Expert 输入 (4 个常规, cached: CPU 写 → BPU 读) ──
        self._ion_expert_state = _alloc(
            'float16', self._expert_in_shapes[0], label="Expert input[0] state")
        self._ion_expert_x_t = _alloc(
            'float16', self._expert_in_shapes[1], label="Expert input[1] x_t")
        self._ion_expert_mask = _alloc(
            'float16', self._expert_in_shapes[2], label="Expert input[2] mask")
        self._ion_expert_pos = _alloc(
            'int32', self._expert_in_shapes[3], label="Expert input[3] pos")

        # ── Expert 输出 ──
        self._ion_expert_out = _alloc(
            'float16', self._expert_out_shapes[0], cached=False,
            label="Expert output[0] actions")

        logger.info(f"  Total ION allocated: {total_ion_bytes / 1024 / 1024:.1f}MB")

    # -----------------------------------------------------------------
    # ION 绑定
    # -----------------------------------------------------------------

    def _bind_models(self):
        """将 ION 绑定到模型的输入/输出。"""
        kv_count = self._expert_kv_count

        # Gemma LLM: 5 inputs + 37 outputs (hidden + 36 KV)
        self._gemma_llm.set_inputs([
            self._ion_gemma_tokens,
            self._ion_gemma_vision,
            self._ion_gemma_mask,
            self._ion_gemma_pos,
            self._ion_gemma_softmax,
        ], n_task=0)
        self._gemma_llm.set_outputs(
            [self._ion_gemma_hidden] + self._ion_kv,
            n_task=0,
        )
        logger.info("  Gemma LLM I/O bound")

        # Expert: 4 inputs + expert_kv_count KV (共享!) + 1 output
        self._expert.set_inputs([
            self._ion_expert_state,
            self._ion_expert_x_t,
            self._ion_expert_mask,
            self._ion_expert_pos,
        ] + self._ion_kv[:kv_count], n_task=0)
        self._expert.set_outputs([self._ion_expert_out], n_task=0)
        logger.info(f"  Expert I/O bound ({kv_count} KV shared with LLM)")

    # -----------------------------------------------------------------
    # 常量初始化
    # -----------------------------------------------------------------

    def _init_constants(self):
        """初始化帧间不变的数据到 ION。"""
        # Expert state: 全零 (Expert 内部不使用)
        self._ion_expert_state.as_array()[:] = 0
        self._ion_expert_state.flush()

        # Expert x_t: 高斯噪声 (seed=42, 与标准版一致)
        rng = np.random.RandomState(42)
        x_t = rng.randn(1, self.action_horizon, self.action_dim).astype(np.float16)
        self._ion_expert_x_t.as_array()[:] = x_t
        self._ion_expert_x_t.flush()

        logger.info("  Constants initialized (state=zeros, x_t=noise)")

    # -----------------------------------------------------------------
    # 推理
    # -----------------------------------------------------------------

    def inference(self, images: dict, state: np.ndarray, text: str) -> np.ndarray:
        """ION 零拷贝推理 + 可选低通滤波。

        异步流水线 + KV Cache 零拷贝 + CPU/BPU 重叠：
          Phase 1: 图像预处理 → 异步提交 siglip (3 实例各绑单核)
          Phase 2: CPU token化 + LLM mask/pos + 写入 LLM ION (与 siglip BPU 并行)
          Phase 3: siglip wait → 写入 vision ION → LLM start
          Phase 3.5: Expert mask/pos 构建 + 写入 Expert ION (与 LLM BPU 并行!)
          Phase 4: LLM wait → Expert start → Expert wait → 后处理 + 可选滤波

        Args:
            images: dict {cam_high, cam_left_wrist, cam_right_wrist}, each HWC uint8
            state: (57,) float32 — raw robot state
            text: prompt string

        Returns:
            np.ndarray (action_horizon, action_dim) float32
        """
        pos = self._siglip_pos

        # ========== Phase 1: SigLIP 异步提交 (3 实例各绑单核, 并行) ==========
        for i, cam_key in enumerate(CAM_KEYS):
            img = preprocess_image(images[cam_key])
            self._siglips[i].safe_start(
                [np.ascontiguousarray(img), pos],
                task_id=0,
            )

        # ========== Phase 2: CPU 工作（与 BPU siglip 并行） ==========
        # -- Tokenize --
        tokens, valid_token_len = tokenize(
            self._tokenizer, text, self._lang_token_count)
        tokens = tokens[np.newaxis, :]  # (1, lang_token_count)

        # -- Gemma LLM mask/pos --
        attn_mask = build_gemma_attn_mask(
            VISION_PREFIX_LEN, valid_token_len, self._lang_token_count)
        pos_ids = make_gemma_position_ids(
            VISION_PREFIX_LEN, valid_token_len, self._lang_token_count)
        softmax_mask = make_softmax_mask(
            VISION_PREFIX_LEN, valid_token_len, self._lang_token_count)

        # -- 写入 LLM ION 输入 (cached, start 时自动 flush) --
        self._ion_gemma_tokens.as_array()[:] = tokens
        self._ion_gemma_mask.as_array()[:] = attn_mask
        self._ion_gemma_pos.as_array()[:] = pos_ids
        self._ion_gemma_softmax.as_array()[:] = softmax_mask

        # ========== Phase 3: siglip wait → LLM → Expert ==========
        # -- 收集 siglip 结果 (3 实例并行, 全部已完成) --
        siglip_outs = []
        for i in range(NUM_VIEWS):
            result = self._siglips[i].wait(task_id=0)
            siglip_outs.append(result[0].astype(np.float32))
        vision_embeds = np.concatenate(siglip_outs, axis=1)  # (1, 768, 2048)
        vision_fp16 = vision_embeds.astype(np.float16)

        # -- 写入 LLM vision ION (cached) --
        self._ion_gemma_vision.as_array()[:] = vision_fp16

        # -- Stage 2: Gemma LLM (KV 写入共享 ION, 零拷贝!) --
        self._gemma_llm.start([], task_id=0)   # flush cached inputs + submit BPU

        # ── 流水线重叠: LLM BPU 执行期间做 Expert 数据准备 ──
        expert_attn_mask = build_expert_attn_mask(
            VISION_PREFIX_LEN, valid_token_len,
            self.action_horizon, self._lang_token_count)
        expert_pos_ids = make_expert_position_ids(
            VISION_PREFIX_LEN, valid_token_len,
            self.action_horizon)
        self._ion_expert_mask.as_array()[:] = expert_attn_mask
        self._ion_expert_pos.as_array()[:] = expert_pos_ids

        self._gemma_llm.wait(task_id=0)         # 等 BPU 完成

        # KV data is now in self._ion_kv — Expert reads from same ION!

        # -- Stage 3: Expert (从共享 ION 读 KV, 零拷贝!) --
        self._expert.start([], task_id=0)        # flush cached inputs + submit BPU
        result = self._expert.wait(task_id=0)    # 等 BPU 完成

        # -- 后处理 --
        actions_raw = result[0].squeeze(0).astype(np.float32)  # (20, 38)
        actions_unnorm = unnormalize_actions(actions_raw, self._norm_stats)

        if self._cam_abs_transform is not None:
            actions_abs = self._cam_abs_transform(
                actions_unnorm, state.astype(np.float32))
        else:
            actions_abs = actions_unnorm

        actions = actions_abs[:, :self.action_dim].astype(np.float32)

        # ── 可选 EMA 帧间平滑 ──
        if self._filter is not None:
            n = self._filter_steps if self._filter_steps else actions.shape[0]
            actions = self._filter.smooth(actions, n)

        return actions