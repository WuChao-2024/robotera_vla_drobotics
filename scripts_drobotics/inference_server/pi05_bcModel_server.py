#!/usr/bin/env python3
"""
Pi05 BC 模型推理服务 — 使用 hbdk4 编译产物 (.convert.bc) 做端到端推理。

协议与 pi05_torch_server.py 完全兼容，可用 compare.py 直接对比。

关键差异（BC vs Torch）：
  - Expert BC 的 build() 将 10 步 flow-matching denoising 写成一个完整 LEAP IR 图
  - 因此 Expert BC 只需一次 feed()，内部自动完成 10 步闭环去噪
  - Expert BC 的 _input_0 (state) 在 build() 中未被使用，填零即可
  - Expert BC 的 _input_3 是 position_ids (int32)，不是 denoise_idx

完整 pipeline（对齐 pi05_oellm_convert.py 的编译管线）：
  输入: HWC uint8 images + raw state(57,) + prompt
    → 图像预处理 (pad→resize→normalize, 纯cv2+numpy, 对齐 Pi05LeapModel)
    → Tokenize (SentencePiece, 对齐 Pi05LeapModel)
    → Stage 1: SigLIP BC (×3, concat) → (1, 768, 2048)
    → Stage 2: Gemma LLM BC → hidden(1, 992, 2048) + 36×KV(1, 992, 256)
    → Stage 3: Gemma Expert BC (一次 feed，10 步闭环去噪已在 IR 中)
    → Action quantile unnormalize
    → CamAbsoluteEeActions (delta→absolute base frame, URDF kinematics)
  输出: actions ndarray (action_horizon, action_dim) float32

环境:
  conda activate robotera_vla_drobotics_convert

用法:
  python pi05_bcModel_server.py --port 8006 --warmup
"""

import argparse
import io
import json
import logging
import os
import time
import traceback

import cv2
import numpy as np
import sentencepiece
import torch
import zmq

from hbdk4.compiler import load as hbdk_load

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pi05_bc")

# =============================================================================
# 默认路径
# =============================================================================

DEFAULT_MODEL_PATH = "/home/chao.wu/pi_oellm/ROBOTREA/M7_pickplace_example_ckpt_pt"
DEFAULT_BC_SIGLIP = "/home/chao.wu/pi_oellm/ROBOTREA/M7_pickplace_example_ckpt_hbm1/siglip/pi05_siglip_pi05_action_horizon_20_ptq.convert.bc"
DEFAULT_BC_GEMMA = "/home/chao.wu/pi_oellm/ROBOTREA/M7_pickplace_example_ckpt_hbm1/gemma_llm/pi05_gemma_llm_pi05_action_horizon_20_ptq.convert.bc"
DEFAULT_BC_EXPERT = "/home/chao.wu/pi_oellm/ROBOTREA/M7_pickplace_example_ckpt_hbm1/gemma_expert/pi05_gemma_expert_pi05_action_horizon_20_ptq.convert.bc"
DEFAULT_URDF_PATH = "/home/chao.wu/pi_oellm/ROBOTREA/robotera_vla_drobotics/l3_4.urdf"
DEFAULT_PROMPT = "pick the apple and put it in the bowl."

# 模型常量（从 config.json 读取或 BC 模型推断）
ACTION_HORIZON = 20
ACTION_DIM = 38
STATE_DIM = 57
VISION_PATCH = 256          # per camera
NUM_VIEWS = 3
VISION_PREFIX_LEN = VISION_PATCH * NUM_VIEWS  # 768
DENOISE_STEPS = 10    # 仅用于文档说明；BC 内部已包含 10 步闭环
NUM_LAYERS = 18

CAM_KEYS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

# =============================================================================
# 图像预处理（从 Pi05LeapModel 复制）
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

def load_tokenizer(model_path: str) -> sentencepiece.SentencePieceProcessor:
    tokenizer_path = os.path.join(model_path, "paligemma_tokenizer.model")
    if not os.path.isfile(tokenizer_path):
        raise FileNotFoundError(f"paligemma_tokenizer.model not found in {model_path}")
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

def load_norm_stats(model_path: str) -> dict:
    path = os.path.join(model_path, "assets", "norm_stats.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"norm_stats.json not found in {model_path}/assets/")
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
    """构建 Gemma LLM additive attention mask (1, 1, seq, seq) float16。

    无效 token 位置设为 float16 最小值的 1/2，避免 NaN。
    """
    seq_len = vision_len + total_lang_len
    mask = np.zeros((1, 1, seq_len, seq_len), dtype=np.float16)
    invalid_lang_len = total_lang_len - valid_lang_len
    if invalid_lang_len > 0:
        invalid_val = np.finfo(np.float16).min / 2  # 避免 softmax NaN
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
# CamAbsoluteEeActions — URDF 后处理（从 Pi05LeapModel 复制）
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
# BC 模型管理器
# =============================================================================

class Pi05BCModel:
    """Pi05 三子模型 BC 推理封装。

    加载三个 .convert.bc 模型，提供 infer() 接口。
    所有预处理和 BC 推理均在 numpy/CPU 上完成。
    """

    def __init__(self,
                 bc_siglip: str,
                 bc_gemma_llm: str,
                 bc_gemma_expert: str,
                 model_path: str,
                 urdf_path: str = None):
        """
        Args:
            bc_siglip: SigLIP .convert.bc 文件路径
            bc_gemma_llm: Gemma LLM .convert.bc 文件路径
            bc_gemma_expert: Gemma Expert .convert.bc 文件路径
            model_path: safetensors 模型目录（含 tokenizer 和 norm_stats）
            urdf_path: URDF 路径（可选，用于 CamAbsoluteEeActions 后处理）
        """
        # 加载 tokenizer 和 norm_stats
        self._tokenizer = load_tokenizer(model_path)
        self._norm_stats = load_norm_stats(model_path)

        # 从 config.json 读取 action_dim, action_horizon
        config_file = os.path.join(model_path, "config.json")
        with open(config_file) as f:
            config = json.load(f)
        self.action_dim = config.get("action_dim", ACTION_DIM)
        self.action_horizon = config.get("action_horizon", ACTION_HORIZON)

        # 加载 BC 模型
        logger.info(f"Loading SigLIP BC: {bc_siglip}")
        t0 = time.time()
        self._siglip_mod = hbdk_load(bc_siglip)
        self._siglip_func = self._siglip_mod.functions[0]
        logger.info(f"  SigLIP loaded in {time.time() - t0:.1f}s")

        logger.info(f"Loading Gemma LLM BC: {bc_gemma_llm}")
        t0 = time.time()
        self._gemma_mod = hbdk_load(bc_gemma_llm)
        self._gemma_func = self._gemma_mod.functions[0]
        logger.info(f"  Gemma LLM loaded in {time.time() - t0:.1f}s")

        logger.info(f"Loading Gemma Expert BC: {bc_gemma_expert}")
        t0 = time.time()
        self._expert_mod = hbdk_load(bc_gemma_expert)
        self._expert_func = self._expert_mod.functions[0]
        logger.info(f"  Gemma Expert loaded in {time.time() - t0:.1f}s")

        # 从 BC 模型输入形状推断参数
        self._infer_bc_params()

        # URDF 后处理
        if urdf_path and os.path.exists(urdf_path):
            self._cam_abs_transform = CamAbsoluteEeActions(urdf_path)
            logger.info(f"URDF transform enabled: {urdf_path}")
        else:
            self._cam_abs_transform = None
            logger.info("URDF transform disabled")

        logger.info(f"BC model ready: action_dim={self.action_dim}, "
                    f"action_horizon={self.action_horizon}, "
                    f"lang_tokens={self._lang_token_count}, "
                    f"total_seq={self._total_seq_len}")

    def _infer_bc_params(self):
        """从 BC 模型输入形状推断 token 数量等参数。"""
        # Gemma LLM 输入: _input_0 = tokens, shape = (1, lang_token_count)
        gemma_inputs = {inp.name: inp for inp in self._gemma_func.flatten_inputs}
        self._lang_token_count = tuple(gemma_inputs["_input_0"].type.shape)[1]
        self._total_seq_len = VISION_PREFIX_LEN + self._lang_token_count

        # Expert 输入: _input_0~_input_3 = (state, x_t, mask, pos_ids)
        #   _input_4 ~ _input_{4+kv_count-1} = KV cache tensors
        expert_inputs = {inp.name: inp for inp in self._expert_func.flatten_inputs}
        self._expert_kv_count = len(expert_inputs) - 4  # 减去前 4 个非 KV 输入
        if self._expert_kv_count != NUM_LAYERS * 2:
            logger.warning(f"Expert KV count={self._expert_kv_count}, expected {NUM_LAYERS * 2}; "
                           f"LLM outputs {NUM_LAYERS * 2} KV tensors. Will use first {self._expert_kv_count} KV tensors.")

    # ---- BC 推理 ----

    def _run_siglip(self, image_chw_fp16: np.ndarray) -> np.ndarray:
        """SigLIP BC 单路推理。

        Args:
            image_chw_fp16: (1, 3, 224, 224) float16

        Returns:
            (1, 256, 2048) float32
        """
        pos = np.arange(0, VISION_PATCH, dtype=np.int64).reshape(1, VISION_PATCH)
        out = self._siglip_func.feed({
            "_input_0": np.ascontiguousarray(image_chw_fp16),
            "_input_1": pos,
        })
        return out["_output_0"].astype(np.float32)

    def _run_gemma_llm(self, tokens: np.ndarray, vision_embeds: np.ndarray,
                       attn_mask: np.ndarray, pos_ids: np.ndarray,
                       softmax_mask: np.ndarray):
        """Gemma LLM BC 推理。

        Args:
            tokens: (1, lang_token_count) int32
            vision_embeds: (1, 768, 2048) float16
            attn_mask: (1, 1, total_seq, total_seq) float16
            pos_ids: (1, total_seq) int32
            softmax_mask: (1, total_seq) float16

        Returns:
            (hidden, kv_list): hidden (1, total_seq, 2048) float32,
                               kv_list = [18 keys + 18 values] float32
        """
        out = self._gemma_func.feed({
            "_input_0": tokens,
            "_input_1": np.ascontiguousarray(vision_embeds),
            "_input_2": np.ascontiguousarray(attn_mask),
            "_input_3": pos_ids,
            "_input_4": np.ascontiguousarray(softmax_mask),
        })
        hidden = out["_output_0"].astype(np.float32)
        kv_list = [out[f"_output_{1 + j}"].astype(np.float32) for j in range(NUM_LAYERS * 2)]
        return hidden, kv_list

    def _run_expert(self, state: np.ndarray, x_t: np.ndarray,
                    attn_mask: np.ndarray, pos_ids: np.ndarray,
                    kv_list: list) -> np.ndarray:
        """Gemma Expert BC 推理 — 一次 feed 包含 10 步闭环去噪。

        Expert BC 的 build() 方法将 10 步 flow-matching denoising 写成一个
        完整的 LEAP IR 计算图，所以一次 feed() 就输出最终去噪结果，
        不需要外部循环。

        输入签名（对齐 get_leap_input_types）：
          _input_0: (1, action_dim) float16      — state（build 中未使用，填零）
          _input_1: (1, horizon, action_dim) f16  — x_t 初始噪声
          _input_2: (1, 1, horizon, token_len) f16 — attention_mask
          _input_3: (1, horizon) int32             — position_ids
          _input_4 ~ _input_{4+35}: 36 × (1, token_len, 256) f16 — KV cache

        Args:
            state: (1, action_dim) float16 — 机器人状态（BC build 中未使用，填零）
            x_t: (1, horizon, action_dim) float16 — 初始高斯噪声
            attn_mask: (1, 1, horizon, token_len) float16
            pos_ids: (1, horizon) int32
            kv_list: 36 个 (1, token_len, 256) float32 — LLM KV cache

        Returns:
            (1, horizon, action_dim) float16 — 10 步去噪后的最终动作
        """
        inputs = {
            "_input_0": np.ascontiguousarray(state),
            "_input_1": np.ascontiguousarray(x_t),
            "_input_2": np.ascontiguousarray(attn_mask),
            "_input_3": pos_ids,
        }
        # KV cache: float32 → float16
        for k in range(len(kv_list)):
            inputs[f"_input_{4 + k}"] = np.ascontiguousarray(kv_list[k].astype(np.float16))
        out = self._expert_func.feed(inputs)
        return out["_output_0"]

    # ---- 完整推理 ----

    def infer(self, images: dict, state: np.ndarray, text: str) -> np.ndarray:
        """完整推理：HWC uint8 images + raw state + prompt → absolute base-frame actions.

        Args:
            images: dict {cam_high, cam_left_wrist, cam_right_wrist}, each HWC uint8
            state: (57,) float32 — raw robot state
            text: prompt string

        Returns:
            np.ndarray (action_horizon, action_dim) float32
        """
        # -- 图像预处理 (numpy) --
        img_tensors = []
        for cam_key in CAM_KEYS:
            img_tensors.append(preprocess_image(images[cam_key]))
        # img_tensors: list of (1, 3, 224, 224) float16

        # -- Tokenize --
        tokens, valid_token_len = tokenize(self._tokenizer, text, self._lang_token_count)
        tokens = tokens[np.newaxis, :]  # (1, lang_token_count) — BC 需要 batch dim

        # ========== Stage 1: SigLIP ==========
        siglip_outs = []
        for img in img_tensors:
            siglip_outs.append(self._run_siglip(img))
        vision_embeds = np.concatenate(siglip_outs, axis=1)  # (1, 768, 2048) float32
        vision_embeds_f16 = vision_embeds.astype(np.float16)  # BC 输入用 float16

        # ========== Stage 2: Gemma LLM ==========
        attn_mask = build_gemma_attn_mask(VISION_PREFIX_LEN, valid_token_len,
                                          self._lang_token_count)
        pos_ids = make_gemma_position_ids(VISION_PREFIX_LEN, valid_token_len,
                                          self._lang_token_count)
        softmax_mask = make_softmax_mask(VISION_PREFIX_LEN, valid_token_len,
                                         self._lang_token_count)

        hidden, kv_list = self._run_gemma_llm(
            tokens, vision_embeds_f16,
            attn_mask, pos_ids, softmax_mask,
        )
        # kv_list: 36 × (1, total_seq, 256) float32, need float16 for Expert

        # ========== Stage 3: Expert — 一次 feed 完成 10 步闭环去噪 ==========
        # Expert BC 的 build() 将 10 步 denoising 写成完整 IR，一次 feed 即出最终结果。
        expert_attn_mask = build_expert_attn_mask(
            VISION_PREFIX_LEN, valid_token_len,
            self.action_horizon, self._lang_token_count,
        )
        expert_pos_ids = make_expert_position_ids(
            VISION_PREFIX_LEN, valid_token_len,
            self.action_horizon,
        )

        # x_t 初始化：高斯噪声 (与 torch server seed=42 对齐)
        torch.manual_seed(42)
        x_t = torch.randn(1, self.action_horizon, self.action_dim,
                         dtype=torch.float32, device='cpu').numpy().astype(np.float16)

        # state 在 Expert BC build() 中未使用，填零向量即可
        action_state = np.zeros((1, self.action_dim), dtype=np.float16)

        # 一次 feed：BC 内部已包含 10 步 x_t += -0.1 * suffix_out 闭环
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


# =============================================================================
# 全局状态
# =============================================================================

_bc_model = None


def load_model(bc_siglip: str, bc_gemma_llm: str, bc_gemma_expert: str,
               model_path: str, urdf_path: str = None):
    global _bc_model
    _bc_model = Pi05BCModel(
        bc_siglip=bc_siglip,
        bc_gemma_llm=bc_gemma_llm,
        bc_gemma_expert=bc_gemma_expert,
        model_path=model_path,
        urdf_path=urdf_path,
    )


def infer(images, state, text):
    return _bc_model.infer(images, state, text)


# =============================================================================
# Warmup
# =============================================================================

def warmup():
    logger.info("Warming up...")
    dummy_images = {
        "cam_high": np.random.randint(0, 256, (480, 848, 3), dtype=np.uint8),
        "cam_left_wrist": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
        "cam_right_wrist": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
    }
    dummy_state = np.random.randn(STATE_DIM).astype(np.float32)

    t0 = time.time()
    actions = infer(dummy_images, dummy_state, DEFAULT_PROMPT)
    elapsed = time.time() - t0
    logger.info(f"Warmup done in {elapsed:.1f}s. Output shape: {actions.shape}")


# =============================================================================
# ZMQ 服务（与 pi05_torch_server.py 协议一致）
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Pi05 BC Model Inference Server")
    p.add_argument("--bc-siglip", type=str, default=DEFAULT_BC_SIGLIP,
                   help="SigLIP .convert.bc path")
    p.add_argument("--bc-gemma-llm", type=str, default=DEFAULT_BC_GEMMA,
                   help="Gemma LLM .convert.bc path")
    p.add_argument("--bc-gemma-expert", type=str, default=DEFAULT_BC_EXPERT,
                   help="Gemma Expert .convert.bc path")
    p.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH,
                   help="safetensors model dir (tokenizer + norm_stats)")
    p.add_argument("--urdf-path", type=str, default=DEFAULT_URDF_PATH,
                   help="URDF path for CamAbsoluteEeActions")
    p.add_argument("--port", type=int, default=45457)
    p.add_argument("--warmup", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    bind_addr = f"tcp://0.0.0.0:{args.port}"

    logger.info("=" * 60)
    logger.info("Pi05 BC Model Inference Server")
    logger.info(f"  SigLIP BC:     {args.bc_siglip}")
    logger.info(f"  Gemma LLM BC:  {args.bc_gemma_llm}")
    logger.info(f"  Expert BC:     {args.bc_gemma_expert}")
    logger.info(f"  Model path:    {args.model_path}")
    logger.info(f"  URDF path:     {args.urdf_path}")
    logger.info(f"  Bind:          {bind_addr}")
    logger.info("=" * 60)

    load_model(args.bc_siglip, args.bc_gemma_llm, args.bc_gemma_expert,
               args.model_path, args.urdf_path)

    if args.warmup:
        warmup()

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(bind_addr)
    logger.info(f"Listening on {bind_addr}...")

    inference_count = 0
    total_time = 0.0

    try:
        while True:
            raw = socket.recv()
            buf = io.BytesIO(raw)
            data = np.load(buf, allow_pickle=True)

            images = data["images"].item() if (
                isinstance(data["images"], np.ndarray)
                and data["images"].dtype == object
            ) else data["images"]
            state = data["state"]
            text_list = data["text"]

            if state.ndim == 1:
                B = 1
                states_batch = state[np.newaxis, :]
                texts_batch = [text_list] if isinstance(text_list, str) else text_list
                images_batch = {
                    k: (v[np.newaxis, ...] if v.ndim == 3 else v)
                    for k, v in images.items()
                }
            else:
                B = state.shape[0]
                states_batch = state
                texts_batch = text_list if isinstance(text_list, list) else [text_list]
                images_batch = images

            all_actions = []
            t_start = time.time()

            for i in range(B):
                img_dict = {k: v[i] if v.ndim == 4 else v
                           for k, v in images_batch.items()}
                s = states_batch[i]
                t = texts_batch[i] if isinstance(texts_batch, list) else texts_batch
                if not t or (isinstance(t, float) and np.isnan(t)):
                    t = DEFAULT_PROMPT

                actions = infer(img_dict, s, str(t))
                all_actions.append(actions)

            elapsed = time.time() - t_start
            total_time += elapsed
            inference_count += 1

            actions_array = np.stack(all_actions, axis=0).astype(np.float32)
            socket.send_pyobj({"actions": actions_array})

            if inference_count % 10 == 0:
                avg = total_time / inference_count
                logger.info(
                    f"[{inference_count}] avg={avg:.3f}s, last={elapsed:.3f}s, "
                    f"shape={actions_array.shape}"
                )

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
    finally:
        socket.close()
        context.term()
        logger.info(f"Server stopped. Total inferences: {inference_count}")


if __name__ == "__main__":
    main()
