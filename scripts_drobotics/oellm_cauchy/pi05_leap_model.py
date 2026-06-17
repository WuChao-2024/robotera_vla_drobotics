#!/usr/bin/env python3
"""
Pi05 LEAP 纯模型模块。

封装三子模型（Siglip / Gemma LLM / Gemma Expert）的加载、tokenizer、
device/dtype 管理和 compile_mode 切换。不包含前向推理、预处理、mask 生成等
管线层逻辑——这些由 server 和 convert 各自处理。

用法:
  from pi05_leap_model import Pi05LeapModel, infer_model_config

  config = infer_model_config("path/to/model.safetensors")
  model = Pi05LeapModel("path/to/model_dir", action_horizon=20, device="cuda")
  model.to(device="cuda", dtype=torch.float16)
  model.compile_mode(False)
  tokens, valid_len = model.tokenize("pick the apple")
"""

import os
import re
import sys

import cv2
import json

import numpy as np

# JAX is lazy-loaded — only imported if preprocess_image_jax() is called
_jax = None


def _lazy_jax():
    """延迟加载 JAX，避免非 JAX 路径产生 JAX 依赖。"""
    global _jax
    if _jax is None:
        import jax
        import jax.numpy as jnp
        _jax = (jax, jnp)
    return _jax
import sentencepiece
import torch

from safetensors import safe_open

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


# =============================================================================
# 从 safetensors 自动推断模型配置
# =============================================================================

def infer_model_config(safetensors_path: str) -> dict:
    """从 safetensors 权重推断所有模型维度。
    Returns:
        dict containing all inferred dimensions
    """
    f = safe_open(safetensors_path, framework="pt")
    # --- Vision (SigLIP) ---
    patch_w = f.get_tensor(
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model."
        "embeddings.patch_embedding.weight"
    )
    vision_hidden_size = patch_w.shape[0]
    patch_size = patch_w.shape[2]
    num_channels = patch_w.shape[1]
    pos_emb = f.get_tensor(
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model."
        "embeddings.position_embedding.weight"
    )
    num_patches = pos_emb.shape[0]
    image_size = int(num_patches ** 0.5) * patch_size
    fc1 = f.get_tensor(
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model."
        "encoder.layers.0.mlp.fc1.weight"
    )
    vision_intermediate_size = fc1.shape[0]
    vision_layers = set()
    for k in f.keys():
        m = re.search(r"vision_tower.*encoder\.layers\.(\d+)\.", k)
        if m:
            vision_layers.add(int(m.group(1)))
    vision_num_hidden_layers = len(vision_layers)
    # --- Projector ---
    proj_w = f.get_tensor(
        "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight"
    )
    projection_dim = proj_w.shape[0]
    # --- Language Model (Gemma) ---
    lm_head_w = f.get_tensor("paligemma_with_expert.paligemma.lm_head.weight")
    vocab_size = lm_head_w.shape[0]
    lm_hidden_size = lm_head_w.shape[1]
    q_proj_l = f.get_tensor(
        "paligemma_with_expert.paligemma.model.language_model."
        "layers.0.self_attn.q_proj.weight"
    )
    k_proj_l = f.get_tensor(
        "paligemma_with_expert.paligemma.model.language_model."
        "layers.0.self_attn.k_proj.weight"
    )
    head_dim = k_proj_l.shape[0]
    lm_num_attention_heads = q_proj_l.shape[0] // head_dim
    lm_num_kv_heads = k_proj_l.shape[0] // head_dim
    gate_proj_l = f.get_tensor(
        "paligemma_with_expert.paligemma.model.language_model."
        "layers.0.mlp.gate_proj.weight"
    )
    lm_intermediate_size = gate_proj_l.shape[0]
    lm_layers = set()
    for k in f.keys():
        m = re.search(r"language_model\.layers\.(\d+)\.", k)
        if m:
            lm_layers.add(int(m.group(1)))
    lm_num_hidden_layers = len(lm_layers)
    # --- Gemma Expert ---
    q_proj_e = f.get_tensor(
        "paligemma_with_expert.gemma_expert.model."
        "layers.0.self_attn.q_proj.weight"
    )
    expert_hidden_size = q_proj_e.shape[1]
    gate_proj_e = f.get_tensor(
        "paligemma_with_expert.gemma_expert.model."
        "layers.0.mlp.gate_proj.weight"
    )
    expert_intermediate_size = gate_proj_e.shape[0]
    expert_layers = set()
    for k in f.keys():
        m = re.search(r"gemma_expert\.model\.layers\.(\d+)\.", k)
        if m:
            expert_layers.add(int(m.group(1)))
    expert_num_hidden_layers = len(expert_layers)
    # --- PI0 Heads ---
    action_in_w = f.get_tensor("action_in_proj.weight")
    action_dim = action_in_w.shape[1]
    config = {
        "vision_hidden_size": vision_hidden_size,
        "vision_intermediate_size": vision_intermediate_size,
        "vision_num_hidden_layers": vision_num_hidden_layers,
        "patch_size": patch_size,
        "num_channels": num_channels,
        "num_patches": num_patches,
        "image_size": image_size,
        "projection_dim": projection_dim,
        "vocab_size": vocab_size,
        "lm_hidden_size": lm_hidden_size,
        "lm_intermediate_size": lm_intermediate_size,
        "lm_num_hidden_layers": lm_num_hidden_layers,
        "lm_num_attention_heads": lm_num_attention_heads,
        "lm_num_kv_heads": lm_num_kv_heads,
        "head_dim": head_dim,
        "expert_hidden_size": expert_hidden_size,
        "expert_intermediate_size": expert_intermediate_size,
        "expert_num_hidden_layers": expert_num_hidden_layers,
        "action_dim": action_dim,
    }
    return config


# =============================================================================
# Pi05LeapModel — 纯模型封装
# =============================================================================

class Pi05LeapModel:
    """Pi05 三子模型加载与管理。

    封装 Siglip、Gemma LLM、Gemma Expert 的 safetensors 加载、
    tokenizer、device/dtype 管理和 compile_mode 切换。
    """

    def __init__(
        self,
        safetensors_path: str,
        action_horizon: int = 20,
        device: str = "cuda",
        use_causal_mask: bool = False,
        use_softmax_mask: bool = False,
        urdf_path: str = None,
    ):
        """
        Args:
            safetensors_path: 包含 model.safetensors 和 tokenizer 的目录路径
            action_horizon: 动作 token 数量
            device: 加载后默认 device
            use_causal_mask: Expert mask 是否使用 causal（Server=False, Convert=False）
            use_softmax_mask: Gemma LLM 是否使用 multiplicative mask
            urdf_path: URDF 路径（仅 server 推理需要，convert 不需要）
        """
        from leap_llm.models.pi05.model_siglip import Siglip
        from leap_llm.models.pi05.model_gemma import LanguageModel
        from leap_llm.models.pi05.model_gemma_expert import GemmaExpertModel

        safetensors_file = os.path.join(safetensors_path, "model.safetensors")
        if not os.path.isfile(safetensors_file):
            raise FileNotFoundError(f"model.safetensors not found in {safetensors_path}")

        self._safetensors_path = safetensors_path
        self._safetensors_file = safetensors_file
        self.action_horizon = action_horizon
        self._use_causal_mask = use_causal_mask
        self._use_softmax_mask = use_softmax_mask

        # 推断模型配置
        self.model_config = infer_model_config(safetensors_file)
        self.action_dim = self.model_config["action_dim"]
        self.vision_token_num = self.model_config["num_patches"]
        print(f"{self.vision_token_num = }")

        # 加载三子模型
        self._siglip = Siglip.build(safetensors_file, self.vision_token_num)
        self._gemma_llm = LanguageModel.build(safetensors_file, self.vision_token_num)
        self._gemma_expert = GemmaExpertModel.build(
            safetensors_file, self.vision_token_num,
            action_horizon=action_horizon,
            action_dim=self.action_dim,
        )

        # 加载 tokenizer
        tokenizer_path = os.path.join(safetensors_path, "paligemma_tokenizer.model")
        with open(tokenizer_path, "rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
        self._max_token_len = 200

        # 默认 device/dtype + eager mode
        self.to(device=device, dtype=torch.float16)
        self.compile_mode(False)

        # 归一化统计
        assets_dir = os.path.join(safetensors_path, "assets")
        self._norm_stats = self._load_norm_stats(assets_dir)

        # URDF 后处理（仅 server 推理需要）
        if urdf_path and os.path.exists(urdf_path):
            self._cam_abs_transform = CamAbsoluteEeActions(urdf_path)
        else:
            self._cam_abs_transform = None

    # ---- 属性 ----

    @property
    def siglip(self):
        return self._siglip

    @property
    def gemma_llm(self):
        return self._gemma_llm

    @property
    def gemma_expert(self):
        return self._gemma_expert

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def max_token_len(self):
        return self._max_token_len

    # ---- Device / dtype ----

    def to(self, device: str = "cuda", dtype: torch.dtype = torch.float16):
        """将三个子模型移到指定 device 和 dtype。"""
        for m in [self._siglip.model, self._gemma_llm.model, self._gemma_expert.model]:
            m.to(device=device, dtype=dtype)

    # ---- Compile mode ----

    def compile_mode(self, mode: bool):
        """统一切换三子模型的 compile_mode。"""
        for m in [self._siglip.model, self._gemma_llm.model, self._gemma_expert.model]:
            m.compile_mode(mode)

    # ---- Tokenize ----

    def tokenize(self, prompt: str):
        """Tokenize 文本并 pad 到 max_token_len。

        Returns:
            (tokens_array, valid_len): np.ndarray (max_len,) of int/bool, int
        """
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        tokens = self._tokenizer.encode(cleaned, add_bos=True) + self._tokenizer.encode("\n")
        tokens_len = len(tokens)
        if tokens_len < self._max_token_len:
            padding = [False] * (self._max_token_len - tokens_len)
            tokens = tokens + padding
        else:
            tokens = tokens[:self._max_token_len]
            tokens_len = self._max_token_len
        return np.asarray(tokens), tokens_len

    # ---- Image preprocessing ----

    @staticmethod
    def _pad_and_resize_to_target(image: np.ndarray, target_height: int,
                                   target_width: int) -> np.ndarray:
        """Pad image to square then resize to target (与 m7_policy.py 一致，cv2)."""
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

    @staticmethod
    def preprocess_image(img_hwc_rgb_uint8: np.ndarray) -> torch.Tensor:
        """图像预处理，精确对齐 torch M7 pi05 推理 pipeline。

        1. cv2 pad + cv2.resize(LINEAR) to 640x480 — 对齐 Torch server M7Inputs
        2. F.interpolate(bilinear, align_corners=False) to 224x224 — 对齐 image_tools.resize_with_pad_torch
        3. uint8 → float32 [-1, 1] — Observation.from_dict: /255.0*2.0-1.0
        4. HWC → CHW

        Args:
            img_hwc_rgb_uint8: np.ndarray [H, W, 3] RGB uint8

        Returns:
            torch.Tensor [1, 3, 224, 224] float32 [-1, 1]
        """
        # Step 1: M7Inputs — pad square + resize to 640x480
        img = Pi05LeapModel._pad_and_resize_to_target(img_hwc_rgb_uint8, 480, 640)

        # Step 2: ResizeImages — torch interpolate 精确对齐 JAX image_tools.resize_with_pad
        cur_h, cur_w = img.shape[:2]
        ratio = max(cur_w / 224, cur_h / 224)
        new_h = int(cur_h / ratio)
        new_w = int(cur_w / ratio)
        img_t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        resized = torch.nn.functional.interpolate(
            img_t, size=(new_h, new_w), mode='bilinear', align_corners=False,
        )
        resized = torch.round(resized).clamp(0, 255).to(torch.uint8)
        pad_h0, rem_h = divmod(224 - new_h, 2)
        pad_w0, rem_w = divmod(224 - new_w, 2)
        padded_img = torch.nn.functional.pad(
            resized, (pad_w0, pad_w0 + rem_w, pad_h0, pad_h0 + rem_h), value=0)
        result = padded_img.squeeze(0).permute(1, 2, 0).numpy()

        # Step 3: normalize to [-1, 1]
        img_tensor = torch.from_numpy(result.astype(np.float32)).unsqueeze(0)
        img_tensor = img_tensor / 255.0 * 2.0 - 1.0

        # Step 4: HWC → CHW
        return img_tensor.permute(0, 3, 1, 2)

    @staticmethod
    def preprocess_image_jax(img_hwc_rgb_uint8: np.ndarray) -> torch.Tensor:
        """图像预处理，对齐 torch 推理 pipeline 的 JAX resize_with_pad。

        1. cv2 pad + cv2.resize(LINEAR) to 640x480 — M7Inputs
        2. jax.image.resize(LINEAR) to 224x224 — image_tools.resize_with_pad
        3. uint8 → float32 [-1, 1] — Observation.from_dict: /255.0*2.0-1.0
        4. HWC → CHW

        Args:
            img_hwc_rgb_uint8: np.ndarray [H, W, 3] RGB uint8

        Returns:
            torch.Tensor [1, 3, 224, 224] float32 [-1, 1]
        """
        # Step 1: M7Inputs — pad square + resize to 640x480
        img = Pi05LeapModel._pad_and_resize_to_target(img_hwc_rgb_uint8, 480, 640)

        # Step 2: JAX resize_with_pad to 224x224 (lazy-loaded JAX)
        jax, jnp = _lazy_jax()
        images = jnp.asarray(img)  # [480, 640, 3] uint8
        cur_h, cur_w = images.shape[0], images.shape[1]
        ratio = max(cur_w / 224, cur_h / 224)
        new_h = int(cur_h / ratio)
        new_w = int(cur_w / ratio)
        resized = jax.image.resize(
            images[None], (1, new_h, new_w, images.shape[2]),
            method=jax.image.ResizeMethod.LINEAR,
        )
        resized = jnp.round(resized).clip(0, 255).astype(jnp.uint8)
        pad_h0, rem_h = divmod(224 - new_h, 2)
        pad_w0, rem_w = divmod(224 - new_w, 2)
        padded = jnp.pad(
            resized,
            ((0, 0), (pad_h0, pad_h0 + rem_h), (pad_w0, pad_w0 + rem_w), (0, 0)),
            constant_values=0,
        )
        result = np.asarray(padded[0])  # [224, 224, 3] uint8

        # Step 3: normalize to [-1, 1]
        img_tensor = torch.from_numpy(result.astype(np.float32)).unsqueeze(0)
        img_tensor = img_tensor / 255.0 * 2.0 - 1.0

        # Step 4: HWC → CHW
        return img_tensor.permute(0, 3, 1, 2)

    # ---- numpy 前处理 (默认, 纯 cv2+numpy, 无 JAX 依赖) ----

    @staticmethod
    def preprocess_image_numpy(img_hwc_rgb_uint8: np.ndarray) -> torch.Tensor:
        """图像预处理 (纯 cv2+numpy)，近似 JAX preprocess_image_jax。

        1. cv2 pad + resize(LINEAR) → 640x480 (squish, 与训练一致)
        2. cv2.resize(INTER_AREA) 等比缩放 + zero-pad → 224x224
        3. normalize → [-1, 1]
        4. HWC → CHW

        步骤 2 用 cv2.INTER_AREA 替代 JAX triangle filter，
        SigLIP 特征 cosine similarity > 0.994。
        """
        # Step 1: 复用 M7Inputs squish
        img = Pi05LeapModel._pad_and_resize_to_target(img_hwc_rgb_uint8, 480, 640)

        # Step 2: cv2.INTER_AREA 等比缩放 + zero-pad to 224x224
        cur_h, cur_w = img.shape[:2]
        ratio = max(cur_w / 224, cur_h / 224)
        new_h, new_w = int(cur_h / ratio), int(cur_w / ratio)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        ph0, rh = divmod(224 - new_h, 2)
        pw0, rw = divmod(224 - new_w, 2)
        padded = np.pad(resized, ((ph0, ph0 + rh), (pw0, pw0 + rw), (0, 0)),
                        mode='constant', constant_values=0)

        # Step 3: normalize to [-1, 1]
        t = torch.from_numpy(padded.astype(np.float32)).unsqueeze(0)
        t = t / 255.0 * 2.0 - 1.0

        # Step 4: HWC → CHW
        return t.permute(0, 3, 1, 2)

    # ---- Mask generation ----

    def build_gemma_mask(self, vision_len: int, valid_lang_len: int,
                         total_lang_len: int = None, device=None, dtype=None):
        """构建 Gemma LLM 的 additive attention mask。"""
        if total_lang_len is None:
            total_lang_len = self._max_token_len
        if device is None:
            device = "cuda"
        if dtype is None:
            dtype = torch.float16

        seq_len = vision_len + total_lang_len
        mask = torch.zeros((1, 1, seq_len, seq_len), dtype=dtype, device=device)
        invalid_lang_len = total_lang_len - valid_lang_len
        if invalid_lang_len > 0:
            invalid_idx = torch.arange(valid_lang_len, total_lang_len, device=device)
            for idx in vision_len + invalid_idx:
                mask[:, :, idx, :] = -32767.0
                mask[:, :, :, idx] = -32767.0
        return mask

    def build_action_expert_mask(self, vision_len: int, valid_prompt_len: int,
                                  action_horizon: int = None, prompt_len: int = None,
                                  device=None, dtype=None):
        """构建 Action Expert 的 attention mask。"""
        if action_horizon is None:
            action_horizon = self.action_horizon
        if prompt_len is None:
            prompt_len = self._max_token_len
        if device is None:
            device = "cuda"
        if dtype is None:
            dtype = torch.float16

        total_cols = vision_len + prompt_len + action_horizon
        mask = torch.zeros((1, 1, action_horizon, total_cols), dtype=dtype, device=device)

        if valid_prompt_len < prompt_len:
            invalid_idx = torch.arange(valid_prompt_len, prompt_len, device=device)
            mask[:, :, :, vision_len + invalid_idx] = -32767.0

        if self._use_causal_mask:
            suffix_start = vision_len + prompt_len
            causal = torch.triu(
                torch.full((action_horizon, action_horizon), -32767.0,
                           dtype=dtype, device=device),
                diagonal=1,
            )
            mask[:, :, :, suffix_start:suffix_start + action_horizon] = causal

        return mask

    # ---- Position IDs ----

    def make_gemma_position_ids(self, vision_token_num: int, valid_prompt_token: int,
                                 total_lang_len: int = None, device=None):
        """生成 Gemma LLM 的位置 ID。"""
        if total_lang_len is None:
            total_lang_len = self._max_token_len
        if device is None:
            device = "cuda"

        total_len = vision_token_num + total_lang_len
        prefix_len = max(0, min(vision_token_num + valid_prompt_token, total_len))
        if prefix_len == 0:
            return torch.zeros((1, total_len), dtype=torch.int32, device=device)
        inc = torch.arange(prefix_len, dtype=torch.int32, device=device)
        tail = torch.full((total_len - prefix_len,), inc[-1].item(),
                          dtype=torch.int32, device=device)
        return torch.cat((inc, tail)).view(1, total_len)

    def make_action_position_ids(self, vision_token_num: int, valid_prompt_token: int,
                                  action_horizon: int = None, device=None):
        """生成 Action Expert 的位置 ID。"""
        if action_horizon is None:
            action_horizon = self.action_horizon
        if device is None:
            device = "cuda"

        start = vision_token_num + valid_prompt_token
        return torch.arange(start, start + action_horizon, dtype=torch.int32,
                            device=device).view(1, action_horizon)

    def make_softmax_mask(self, vision_token_num: int, valid_prompt_token: int,
                           total_lang_len: int = None, device=None):
        """生成 multiplicative softmax mask（仅 convert 模式使用）。"""
        if total_lang_len is None:
            total_lang_len = self._max_token_len
        if device is None:
            device = "cuda"

        total_len = vision_token_num + total_lang_len
        mask = torch.zeros((1, total_len), dtype=torch.float32, device=device)
        mask[:, :vision_token_num + valid_prompt_token] = 1.0
        return mask

    # ---- Forward pass ----

    def forward(self, images_tensor_list, prompt_tokens, valid_token_len,
                action_horizon=None, seed=None):
        """完整前向推理：Siglip → concat → Gemma LLM → Gemma Expert 10-step denoising。

        Args:
            images_tensor_list: list of [1, 3, 224, 224] tensors（已预处理、已置 device/dtype）
            prompt_tokens: np.ndarray of padded token IDs
            valid_token_len: int — 有效文本 token 数量
            action_horizon: int，默认使用模型自身的 action_horizon
            seed: int，随机种子（用于 x_t 初始噪声）

        Returns:
            x_t: torch.Tensor [1, action_horizon, action_dim] — 模型原始输出
        """
        if action_horizon is None:
            action_horizon = self.action_horizon

        device = images_tensor_list[0].device
        dtype = images_tensor_list[0].dtype

        # 1. Siglip 视觉编码 → concat embeddings
        siglip_pos_ids = torch.arange(0, 256).view(1, 256).to(device)
        siglip_outputs = []
        for img in images_tensor_list:
            siglip_outputs.append(self._siglip.model.forward(img, siglip_pos_ids))

        inputs_embeds = torch.concat(siglip_outputs, dim=1)
        vision_token_len = inputs_embeds.shape[1]

        # 2. 文本 token → tensor
        lang_token_t = torch.from_numpy(
            prompt_tokens.astype(np.int32)
        ).unsqueeze(0).to(device)

        # 3. Gemma LLM prefix → KV cache
        attention_mask = self.build_gemma_mask(
            vision_token_len, valid_token_len, device=device, dtype=dtype,
        )
        gemma_pos_ids = self.make_gemma_position_ids(
            vision_token_len, valid_token_len, device=device,
        )
        softmax_mask = (
            self.make_softmax_mask(vision_token_len, valid_token_len, device=device)
            if self._use_softmax_mask else None
        )

        gemma_outputs = self._gemma_llm.model.forward(
            tokens=lang_token_t,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=gemma_pos_ids,
            softmax_mask=softmax_mask,
        )
        kv_cache = gemma_outputs[1:]

        # 4. Action Expert 10-step flow-matching denoising
        action_mask = self.build_action_expert_mask(
            vision_token_len, valid_token_len,
            action_horizon=action_horizon, device=device, dtype=dtype,
        )
        act_pos_ids = self.make_action_position_ids(
            vision_token_len, valid_token_len,
            action_horizon=action_horizon, device=device,
        )

        if seed is not None:
            torch.manual_seed(seed)
        x_t = torch.randn(1, action_horizon, self.action_dim, device=device, dtype=dtype)
        action_state = torch.zeros(1, self.action_dim, device=device, dtype=dtype)

        dt = -1.0 / 10
        for step in range(10):
            denoise_idx = torch.tensor([step], dtype=torch.int32)
            v_t = self._gemma_expert.model.forward(
                state=action_state,
                x_t=x_t,
                denoise_idx=denoise_idx,
                attention_mask=action_mask,
                position_ids=act_pos_ids,
                caches=kv_cache,
            )
            x_t = x_t + dt * v_t

        return x_t

    # ---- Norm stats ----

    @staticmethod
    def _load_norm_stats(assets_dir: str) -> dict:
        path = os.path.join(assets_dir, "norm_stats.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)["norm_stats"]

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Quantile normalize: (x - q01) / (q99 - q01) * 2 - 1."""
        ns = self._norm_stats
        if ns is None or "state" not in ns:
            return state
        s = ns["state"]
        q01 = np.array(s["q01"], dtype=np.float32)[:state.shape[-1]]
        q99 = np.array(s["q99"], dtype=np.float32)[:state.shape[-1]]
        return (state - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

    def _unnormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """Inverse quantile normalize: (x + 1) / 2 * (q99 - q01) + q01."""
        ns = self._norm_stats
        if ns is None or "actions" not in ns:
            return actions
        a = ns["actions"]
        q01 = np.array(a["q01"], dtype=np.float32)[:actions.shape[-1]]
        q99 = np.array(a["q99"], dtype=np.float32)[:actions.shape[-1]]
        return (actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01

    # ---- Full inference (server) ----

    def infer(self, images: dict, state: np.ndarray, text: str) -> np.ndarray:
        """完整推理：HWC uint8 images + raw state + prompt → absolute base-frame actions.

        Args:
            images: dict {cam_high, cam_left_wrist, cam_right_wrist}, each HWC uint8
            state: (57,) float32 — raw robot state
            text: prompt string

        Returns:
            np.ndarray (action_horizon, action_dim) float32
        """
        # Determine device/dtype from model parameters
        p = next(self._siglip.model.parameters())
        device = p.device
        dtype = p.dtype
        cam_keys = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

        # 图像预处理
        images_tensors = []
        for cam_key in cam_keys:
            img_tensor = self.preprocess_image_numpy(images[cam_key])
            images_tensors.append(img_tensor.to(device).to(dtype=dtype))

        # Tokenize
        tokens, valid_token_len = self.tokenize(text)

        # 模型前向
        x_t = self.forward(images_tensors, tokens, valid_token_len, seed=42)
        actions_raw = x_t.squeeze(0).cpu().float().numpy()

        # 后处理
        state_raw = state.astype(np.float32)
        actions_unnorm = self._unnormalize_actions(actions_raw)
        if self._cam_abs_transform is not None:
            actions_abs = self._cam_abs_transform(actions_unnorm, state_raw)
        else:
            actions_abs = actions_unnorm

        return actions_abs[:, :self.action_dim].astype(np.float32)

    # ---- HBM 编译 ----

    def compile_siglip(self, output_model_path: str, **compile_kwargs):
        """编译 Siglip 为 HBM。"""
        self._siglip.compile(
            output_model_path=output_model_path,
            enable_vpu=True,
            enable_spu=False,
            **compile_kwargs,
        )

    def compile_gemma_llm(self, output_model_path: str, **compile_kwargs):
        """编译 Gemma LLM 为 HBM。"""
        self._gemma_llm.compile(
            output_model_path=output_model_path,
            enable_vpu=True,
            **compile_kwargs,
        )

    def compile_gemma_expert(self, output_model_path: str, **compile_kwargs):
        """编译 Gemma Expert 为 HBM。"""
        self._gemma_expert.compile(
            output_model_path=output_model_path,
            enable_vpu=True,
            **compile_kwargs,
        )



# =============================================================================
# CamAbsoluteEeActions — URDF 后处理（预处理 → 前向 → 后处理）
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

