#!/usr/bin/env python3
"""
BPU Pi05 VLA 模型推理封装 — pyCauchyKesai 非零拷贝接口。

提供 BPU_ROBOTREA_VLA_Pi05 类，封装完整的 Pi05 视觉-语言-动作 (VLA)
三阶段推理管线：

  输入: HWC uint8 images + raw state(57,) + prompt
    → 图像预处理 (pad→resize→normalize, 纯cv2+numpy)
    → Tokenize (SentencePiece)
    → Stage 1: SigLIP HBM (×3, concat) → (1, 768, 2048)
    → Stage 2: Gemma LLM HBM → hidden(1, 992, 2048) + 36×KV(1, 992, 256)
    → Stage 3: Gemma Expert HBM (一次 inference，10 步去噪已在 IR 中)
    → Action quantile unnormalize
    → CamAbsoluteEeActions (delta→absolute base frame, URDF kinematics)
  输出: actions ndarray (action_horizon, action_dim) float32

用法:
  from bpu_robotrea_pi05model import BPU_ROBOTREA_VLA_Pi05

  model = BPU_ROBOTREA_VLA_Pi05(
      model_dir="/root/ssd/OELLM_Runtime/robotrea_model/v1",
      share_dir="/root/ssd/OELLM_Runtime/robotrea_model/share",
  )
  actions = model.inference(images_dict, state_array, "pick the apple")
"""

import glob
import json
import logging
import os
import time

import cv2
import numpy as np
import sentencepiece

# L2 Cache 必须在 import pyCauchyKesai 之前设置
os.environ["HB_DNN_USER_DEFINED_L2M_SIZES"] = "6:6:6:6"

from pyCauchyKesai import CauchyKesai

logger = logging.getLogger("bpu_robotrea_pi05")

# =============================================================================
# 模型常量
# =============================================================================

ACTION_HORIZON = 20
ACTION_DIM = 38
STATE_DIM = 57
VISION_PATCH = 256          # per camera
NUM_VIEWS = 3
VISION_PREFIX_LEN = VISION_PATCH * NUM_VIEWS  # 768
DENOISE_STEPS = 10    # 仅用于文档说明；HBM 内部已包含 10 步闭环
NUM_LAYERS = 18

CAM_KEYS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


# =============================================================================
# 图像预处理（对齐 Pi05LeapModel / M7 训练管线）
# =============================================================================

def _pad_and_resize_to_target(image: np.ndarray, target_height: int,
                              target_width: int) -> np.ndarray:
    """Pad image to square then resize to target (cv2, 与 M7 训练一致)."""
    h, w = image.shape[:2]
    if h > w:
        pad_total = h - w
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        pad_top = 0
        pad_bottom = 0
    else:
        pad_total = w - h
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        pad_left = 0
        pad_right = 0
    padded = cv2.copyMakeBorder(
        image, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=[0, 0, 0],
    )
    return cv2.resize(padded, (target_width, target_height),
                      interpolation=cv2.INTER_LINEAR)


def preprocess_image(img_hwc_rgb_uint8: np.ndarray) -> np.ndarray:
    """图像预处理，精确对齐 torch M7 pi05 推理 pipeline。

    1. cv2 pad + cv2.resize(LINEAR) to 640x480 — M7Inputs
    2. cv2.resize(INTER_AREA) 等比缩放 + zero-pad to 224x224
    3. uint8 → float32 [-1, 1] → float16
    4. HWC → CHW

    Args:
        img_hwc_rgb_uint8: np.ndarray [H, W, 3] RGB uint8

    Returns:
        np.ndarray [1, 3, 224, 224] float16 [-1, 1]
    """
    # Step 1: M7Inputs — pad square + resize to 640x480
    img = _pad_and_resize_to_target(img_hwc_rgb_uint8, 480, 640)  # RGB

    # Step 2: cv2.INTER_AREA 等比缩放 + zero-pad to 224x224
    cur_h, cur_w = img.shape[:2]
    ratio = max(cur_w / 224, cur_h / 224)
    new_h, new_w = int(cur_h / ratio), int(cur_w / ratio)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    ph0, rh = divmod(224 - new_h, 2)
    pw0, rw = divmod(224 - new_w, 2)
    padded = np.pad(resized, ((ph0, ph0 + rh), (pw0, pw0 + rw), (0, 0)),
                    mode='constant', constant_values=0)

    # Step 3: normalize to [-1, 1], cast to float16
    t = padded.astype(np.float32)
    t = t / 255.0 * 2.0 - 1.0

    # Step 4: HWC → CHW, add batch dim
    return np.ascontiguousarray(t.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float16))


# =============================================================================
# Tokenizer
# =============================================================================

def load_tokenizer(share_dir: str) -> sentencepiece.SentencePieceProcessor:
    tokenizer_path = os.path.join(share_dir, "paligemma_tokenizer.model")
    if not os.path.isfile(tokenizer_path):
        raise FileNotFoundError(f"paligemma_tokenizer.model not found in {share_dir}")
    with open(tokenizer_path, "rb") as f:
        return sentencepiece.SentencePieceProcessor(model_proto=f.read())


def tokenize(sp: sentencepiece.SentencePieceProcessor, prompt: str, max_token_len: int):
    """Tokenize 文本并 pad 到 max_token_len。

    Returns:
        (tokens_array, valid_len): np.ndarray (max_len,) int32, int
    """
    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    tokens = sp.encode(cleaned, add_bos=True) + sp.encode("\n")
    valid_len = len(tokens)
    if valid_len < max_token_len:
        tokens = tokens + [0] * (max_token_len - valid_len)
    else:
        tokens = tokens[:max_token_len]
        valid_len = max_token_len
    return np.asarray(tokens, dtype=np.int32), valid_len


# =============================================================================
# Norm stats
# =============================================================================

def load_norm_stats(share_dir: str) -> dict:
    path = os.path.join(share_dir, "norm_stats.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"norm_stats.json not found in {share_dir}")
    with open(path) as f:
        return json.load(f)["norm_stats"]


def normalize_state(state: np.ndarray, norm_stats: dict) -> np.ndarray:
    """Quantile normalize: (x - q01) / (q99 - q01) * 2 - 1."""
    ns = norm_stats["state"]
    q01 = np.array(ns["q01"], dtype=np.float32)[:state.shape[-1]]
    q99 = np.array(ns["q99"], dtype=np.float32)[:state.shape[-1]]
    return (state - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def unnormalize_actions(actions: np.ndarray, norm_stats: dict) -> np.ndarray:
    """Inverse quantile normalize: (x + 1) / 2 * (q99 - q01) + q01."""
    ns = norm_stats["actions"]
    q01 = np.array(ns["q01"], dtype=np.float32)[:actions.shape[-1]]
    q99 = np.array(ns["q99"], dtype=np.float32)[:actions.shape[-1]]
    return (actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


# =============================================================================
# Mask / Position ID 生成（numpy 版本，对齐 Pi05LeapModel）
# =============================================================================

def build_gemma_attn_mask(vision_len: int, valid_lang_len: int,
                          total_lang_len: int) -> np.ndarray:
    """构建 Gemma LLM additive attention mask (1, 1, seq, seq) float16。"""
    seq_len = vision_len + total_lang_len
    mask = np.zeros((1, 1, seq_len, seq_len), dtype=np.float16)
    invalid_lang_len = total_lang_len - valid_lang_len
    if invalid_lang_len > 0:
        invalid_val = np.finfo(np.float16).min / 2
        for idx in range(valid_lang_len, total_lang_len):
            abs_idx = vision_len + idx
            mask[:, :, abs_idx, :] = invalid_val
            mask[:, :, :, abs_idx] = invalid_val
    return mask


def make_gemma_position_ids(vision_token_num: int, valid_prompt_token: int,
                             total_lang_len: int) -> np.ndarray:
    """Gemma LLM position IDs: [0,1,2,...,prefix-1, prefix-1, ...] (1, seq) int32."""
    total_len = vision_token_num + total_lang_len
    prefix_len = vision_token_num + valid_prompt_token
    if prefix_len == 0:
        return np.zeros((1, total_len), dtype=np.int32)
    inc = np.arange(prefix_len, dtype=np.int32)
    tail = np.full(total_len - prefix_len, inc[-1], dtype=np.int32)
    return np.concatenate([inc, tail]).reshape(1, total_len)


def make_softmax_mask(vision_token_num: int, valid_prompt_token: int,
                       total_lang_len: int) -> np.ndarray:
    """Multiplicative softmax mask: 前 prefix_len 为 1，其余为 0 (1, seq) float16."""
    total_len = vision_token_num + total_lang_len
    mask = np.zeros((1, total_len), dtype=np.float16)
    mask[:, :vision_token_num + valid_prompt_token] = 1.0
    return mask


def build_expert_attn_mask(vision_len: int, valid_prompt_len: int,
                            action_horizon: int, prompt_len: int) -> np.ndarray:
    """构建 Action Expert attention mask (1, 1, action_horizon, total_cols) float16。"""
    total_cols = vision_len + prompt_len + action_horizon
    mask = np.zeros((1, 1, action_horizon, total_cols), dtype=np.float16)
    invalid_val = np.finfo(np.float16).min / 2
    if valid_prompt_len < prompt_len:
        for idx in range(valid_prompt_len, prompt_len):
            mask[:, :, :, vision_len + idx] = invalid_val
    return mask


def make_expert_position_ids(vision_token_num: int, valid_prompt_token: int,
                              action_horizon: int) -> np.ndarray:
    """Action Expert position IDs: [start, start+1, ..., start+horizon-1] (1, horizon) int32."""
    start = vision_token_num + valid_prompt_token
    return np.arange(start, start + action_horizon, dtype=np.int32).reshape(1, action_horizon)


# =============================================================================
# CamAbsoluteEeActions — URDF 后处理
# =============================================================================

class CamAbsoluteEeActions:
    """将末端空间 delta 动作转为绝对动作（camera frame → base frame），使用 URDF 正运动学。"""

    def __init__(self, urdf_path: str):
        import pinocchio as pin
        from scipy.spatial.transform import Rotation as R

        self.R = R
        self.pin = pin
        self.urdf_model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.urdf_model.createData()
        self.qpos = np.zeros(self.urdf_model.nq)
        self.rs_frame_id = self.urdf_model.getFrameId("stereo_link")
        self.base_frame_id = self.urdf_model.getFrameId("base_link")

    @staticmethod
    def _quat_pos_to_homogeneous(quat_pos: np.ndarray) -> np.ndarray:
        """(n,7) [x,y,z,qx,qy,qz,qw] → (n,4,4) homogeneous."""
        from scipy.spatial.transform import Rotation as R
        n = quat_pos.shape[0]
        H = np.zeros((n, 4, 4))
        H[:, 3, 3] = 1.0
        H[:, :3, 3] = quat_pos[:, :3]
        H[:, :3, :3] = R.from_quat(quat_pos[:, 3:7]).as_matrix()
        return H

    @staticmethod
    def _homogeneous_to_quat_pos(H: np.ndarray) -> np.ndarray:
        """(n,4,4) → (n,7) [x,y,z,qx,qy,qz,qw]."""
        from scipy.spatial.transform import Rotation as R
        n = H.shape[0]
        qp = np.zeros((n, 7))
        qp[:, :3] = H[:, :3, 3]
        qp[:, 3:] = R.from_matrix(H[:, :3, :3]).as_quat()
        return qp

    def _get_rs_frame(self, waist_qpos: np.ndarray, neck_qpos: np.ndarray) -> np.ndarray:
        """Compute T_base_rs (base→stereo camera transform) from waist & neck joints."""
        self.qpos[13:15] = waist_qpos[:2]
        self.qpos[12] = waist_qpos[2]
        self.qpos[34:36] = neck_qpos
        self.pin.forwardKinematics(self.urdf_model, self.data, self.qpos)
        self.pin.framesForwardKinematics(self.urdf_model, self.data, self.qpos)
        T_world_base = self.data.oMf[self.base_frame_id]
        T_world_rs = self.data.oMf[self.rs_frame_id]
        T_base_rs = T_world_base.inverse() * T_world_rs
        return T_base_rs.homogeneous

    def __call__(self, actions: np.ndarray, state: np.ndarray) -> np.ndarray:
        """delta camera-frame actions → absolute base-frame actions."""
        actions = actions.copy()
        state = state.copy()

        T_base_rs = self._get_rs_frame(state[52:55], state[55:57])

        state_rs = state.copy()
        for state_slice in [slice(14, 21), slice(21, 28)]:
            T_base_arm = self._quat_pos_to_homogeneous(state[state_slice].reshape(1, -1))
            state_rs[state_slice] = self._homogeneous_to_quat_pos(
                np.linalg.inv(T_base_rs) @ T_base_arm
            ).reshape(-1)

        pairs = [
            ([0, 1, 2], [3, 4, 5, 6], [14, 15, 16], [17, 18, 19, 20]),
            ([19, 20, 21], [22, 23, 24, 25], [21, 22, 23], [24, 25, 26, 27]),
        ]
        for a_xyz, a_quat, s_xyz, s_quat in pairs:
            actions[:, a_xyz] += state_rs[s_xyz]
            R_state_rs = self.R.from_quat(state_rs[s_quat]).as_matrix()
            R_relative = self.R.from_quat(actions[:, a_quat]).as_matrix()
            actions[:, a_quat] = self.R.from_matrix(R_state_rs @ R_relative).as_quat()

        for action_slice in [slice(0, 7), slice(19, 26)]:
            T_rs_arm = self._quat_pos_to_homogeneous(actions[:, action_slice])
            T_base_arm = T_base_rs @ T_rs_arm
            actions[:, action_slice] = self._homogeneous_to_quat_pos(T_base_arm)

        return actions


# =============================================================================
# BPU_ROBOTREA_VLA_Pi05 — 主类
# =============================================================================

class BPU_ROBOTREA_VLA_Pi05:
    """Pi05 VLA 模型 BPU 推理封装（pyCauchyKesai 非零拷贝接口）。

    加载三个 .hbm 模型（SigLIP + Gemma LLM + Gemma Expert），
    inference() 方法封装完整管线：预处理 → 三阶段推理 → 后处理。

    pyCauchyKesai 非零拷贝模式:
      - CauchyKesai(model_path, n_task=1, model_cnt_select=0) 构造
      - inference([input0, input1, ...]) 接受 list 输入
      - 返回 list 输出: result[0], result[1], ...
      - 不使用 IONArray，numpy 数组自动拷贝到 BPU
    """

    def __init__(self,
                 siglip_path: str,
                 gemma_llm_path: str,
                 expert_path: str,
                 share_dir: str,
                 urdf_path: str = None):
        """
        Args:
            siglip_path: SigLIP .hbm 模型文件路径
            gemma_llm_path: Gemma LLM .hbm 模型文件路径
            expert_path: Gemma Expert .hbm 模型文件路径
            share_dir: 共享资源目录（含 tokenizer、norm_stats、config.json）
            urdf_path: URDF 路径（可选，用于 CamAbsoluteEeActions 后处理）
        """
        # 加载 tokenizer 和 norm_stats
        self._tokenizer = load_tokenizer(share_dir)
        self._norm_stats = load_norm_stats(share_dir)

        # 从 config.json 读取 action_dim, action_horizon
        config_file = os.path.join(share_dir, "config.json")
        with open(config_file) as f:
            config = json.load(f)
        self.action_dim = config.get("action_dim", ACTION_DIM)
        self.action_horizon = config.get("action_horizon", ACTION_HORIZON)

        # HBM 模型路径
        self._siglip_path = siglip_path
        self._gemma_llm_path = gemma_llm_path
        self._expert_path = expert_path

        logger.info(f"Loading Gemma Expert HBM: {self._expert_path}")
        t0 = time.time()
        self._expert = CauchyKesai(self._expert_path, n_task=1, model_cnt_select=0)
        self._expert.set_scheduling_params(bpu_cores=[0, 1, 2, 3])  # 4 核模型
        logger.info(f"  Gemma Expert loaded in {time.time() - t0:.1f}s")
        
        logger.info(f"Loading Gemma LLM HBM: {self._gemma_llm_path}")
        t0 = time.time()
        self._gemma_llm = CauchyKesai(self._gemma_llm_path, n_task=1, model_cnt_select=0)
        self._gemma_llm.set_scheduling_params(bpu_cores=[0, 1, 2, 3])  # 4 核模型
        logger.info(f"  Gemma LLM loaded in {time.time() - t0:.1f}s")
        
        # 加载 HBM 模型 (pyCauchyKesai 非零拷贝模式)
        logger.info(f"Loading SigLIP HBM: {self._siglip_path}")
        t0 = time.time()
        self._siglip = CauchyKesai(self._siglip_path, n_task=3, model_cnt_select=0)
        # siglip 单核模型，n_task=3 支持三路图像异步推理
        self._siglip_pos = np.arange(0, VISION_PATCH, dtype=np.int64).reshape(1, VISION_PATCH)
        logger.info(f"  SigLIP loaded in {time.time() - t0:.1f}s (n_task=3)")





        # 从 HBM 模型输入形状推断参数
        self._infer_model_params()

        # URDF 后处理
        if urdf_path and os.path.exists(urdf_path):
            self._cam_abs_transform = CamAbsoluteEeActions(urdf_path)
            logger.info(f"URDF transform enabled: {urdf_path}")
        else:
            self._cam_abs_transform = None
            logger.info("URDF transform disabled")

        logger.info(f"BPU_ROBOTREA_VLA_Pi05 ready: action_dim={self.action_dim}, "
                    f"action_horizon={self.action_horizon}, "
                    f"lang_tokens={self._lang_token_count}, "
                    f"total_seq={self._total_seq_len}")

    def _infer_model_params(self):
        """从 HBM 模型输入形状推断 token 数量等参数。"""
        gemma_info = self._gemma_llm.s()
        expert_info = self._expert.s()

        # Gemma LLM input[0] = tokens, shape = (1, lang_token_count)
        gemma_inputs = gemma_info["inputs"]
        self._lang_token_count = gemma_inputs[0]["shape"][1]
        self._total_seq_len = VISION_PREFIX_LEN + self._lang_token_count

        # Expert KV cache 数量 = 总输入数 - 4 (state, x_t, mask, pos_ids)
        expert_inputs = expert_info["inputs"]
        self._expert_kv_count = len(expert_inputs) - 4
        if self._expert_kv_count != NUM_LAYERS * 2:
            logger.warning(
                f"Expert KV count={self._expert_kv_count}, "
                f"expected {NUM_LAYERS * 2}; "
                f"LLM outputs {NUM_LAYERS * 2} KV tensors. "
                f"Will use first {self._expert_kv_count} KV tensors.")

        # Expert KV cache seq_len (从 input[4] shape 推断)
        self._expert_kv_seq_len = expert_inputs[4]["shape"][1]

        logger.info(f"  lang_token_count={self._lang_token_count}")
        logger.info(f"  total_seq_len={self._total_seq_len}")
        logger.info(f"  expert_kv_count={self._expert_kv_count}")
        logger.info(f"  expert_kv_seq_len={self._expert_kv_seq_len}")

    # ---- BPU 推理（内部方法） ----

    def _run_siglip(self, image_chw_fp16: np.ndarray) -> np.ndarray:
        """SigLIP HBM 单路推理。

        Args:
            image_chw_fp16: (1, 3, 224, 224) float16

        Returns:
            (1, 256, 2048) float32
        """
        pos = np.arange(0, VISION_PATCH, dtype=np.int64).reshape(1, VISION_PATCH)
        result = self._siglip.inference([
            np.ascontiguousarray(image_chw_fp16),
            pos,
        ])
        return result[0].astype(np.float32)

    def _run_gemma_llm(self, tokens: np.ndarray, vision_embeds: np.ndarray,
                       attn_mask: np.ndarray, pos_ids: np.ndarray,
                       softmax_mask: np.ndarray):
        """Gemma LLM HBM 推理。

        Returns:
            (hidden, kv_list): hidden (1, total_seq, 2048) float32,
                               kv_list = [36 KV tensors] float32
        """
        result = self._gemma_llm.inference([
            tokens,                                 # _input_0
            np.ascontiguousarray(vision_embeds),    # _input_1
            np.ascontiguousarray(attn_mask),        # _input_2
            pos_ids,                                # _input_3
            np.ascontiguousarray(softmax_mask),     # _input_4
        ])
        hidden = result[0].astype(np.float32)
        kv_list = [result[1 + j].astype(np.float32) for j in range(NUM_LAYERS * 2)]
        return hidden, kv_list

    def _run_expert(self, state: np.ndarray, x_t: np.ndarray,
                    attn_mask: np.ndarray, pos_ids: np.ndarray,
                    kv_list: list) -> np.ndarray:
        """Gemma Expert HBM 推理 — 一次 inference 包含 10 步闭环去噪。

        Returns:
            (1, horizon, action_dim) float16 — 10 步去噪后的最终动作
        """
        inputs = [
            np.ascontiguousarray(state),        # _input_0
            np.ascontiguousarray(x_t),           # _input_1
            np.ascontiguousarray(attn_mask),     # _input_2
            pos_ids,                             # _input_3
        ]
        # KV cache: float32 → float16
        for k in range(len(kv_list)):
            inputs.append(np.ascontiguousarray(kv_list[k].astype(np.float16)))

        result = self._expert.inference(inputs)
        return result[0]

    # ---- 完整推理（异步流水线优化） ----

    def inference(self, images: dict, state: np.ndarray, text: str) -> np.ndarray:
        """完整推理：HWC uint8 images + raw state + prompt → absolute base-frame actions.

        异步流水线：
          Phase 1: 图像预处理 → 异步提交 siglip (3路 task_id=0,1,2)
          Phase 2: token化 + 所有 mask 构建 + expert 预处理 (CPU，与 siglip BPU 并行)
          Phase 3: wait siglip → concat → gemma_llm → expert → 后处理

        Args:
            images: dict {cam_high, cam_left_wrist, cam_right_wrist}, each HWC uint8
            state: (57,) float32 — raw robot state
            text: prompt string

        Returns:
            np.ndarray (action_horizon, action_dim) float32
        """
        pos = self._siglip_pos

        # ========== Phase 1: 图像预处理 + 异步提交 siglip ==========
        for i, cam_key in enumerate(CAM_KEYS):
            img = preprocess_image(images[cam_key])
            self._siglip.safe_start(
                [np.ascontiguousarray(img), pos],
                task_id=i,
            )

        # ========== Phase 2: CPU 工作（与 BPU siglip 并行） ==========
        # -- Tokenize --
        tokens, valid_token_len = tokenize(self._tokenizer, text, self._lang_token_count)
        tokens = tokens[np.newaxis, :]  # (1, lang_token_count)

        # -- Gemma LLM 所需的 mask/pos --
        attn_mask = build_gemma_attn_mask(VISION_PREFIX_LEN, valid_token_len,
                                          self._lang_token_count)
        pos_ids = make_gemma_position_ids(VISION_PREFIX_LEN, valid_token_len,
                                          self._lang_token_count)
        softmax_mask = make_softmax_mask(VISION_PREFIX_LEN, valid_token_len,
                                         self._lang_token_count)

        # -- Expert 所需的 mask/pos/x_t（不依赖 gemma_llm 输出） --
        expert_attn_mask = build_expert_attn_mask(
            VISION_PREFIX_LEN, valid_token_len,
            self.action_horizon, self._lang_token_count,
        )
        expert_pos_ids = make_expert_position_ids(
            VISION_PREFIX_LEN, valid_token_len,
            self.action_horizon,
        )

        # x_t 初始化：高斯噪声 (固定 seed=42)
        rng = np.random.RandomState(42)
        x_t = rng.randn(1, self.action_horizon, self.action_dim).astype(np.float16)

        # state 在 Expert 中未使用，填零向量
        action_state = np.zeros((1, self.action_dim), dtype=np.float16)

        # ========== Phase 3: 收集 siglip → gemma_llm → expert ==========
        # -- 收集 siglip 结果 --
        siglip_outs = []
        for i in range(NUM_VIEWS):
            result = self._siglip.wait(task_id=i)
            siglip_outs.append(result[0].astype(np.float32))
        vision_embeds = np.concatenate(siglip_outs, axis=1)  # (1, 768, 2048)
        vision_embeds_f16 = vision_embeds.astype(np.float16)

        # -- Stage 2: Gemma LLM --
        hidden, kv_list = self._run_gemma_llm(
            tokens, vision_embeds_f16,
            attn_mask, pos_ids, softmax_mask,
        )

        # -- Stage 3: Expert — 一次 inference 完成 10 步闭环去噪 --
        x_t_final = self._run_expert(
            action_state, x_t,
            expert_attn_mask, expert_pos_ids,
            kv_list,
        )

        # -- 后处理 --
        actions_raw = x_t_final.squeeze(0).astype(np.float32)  # (20, 38)
        actions_unnorm = unnormalize_actions(actions_raw, self._norm_stats)

        if self._cam_abs_transform is not None:
            actions_abs = self._cam_abs_transform(actions_unnorm, state.astype(np.float32))
        else:
            actions_abs = actions_unnorm

        return actions_abs[:, :self.action_dim].astype(np.float32)
