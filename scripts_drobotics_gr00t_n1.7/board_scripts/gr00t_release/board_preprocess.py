"""板端 text → input_embeds/position_ids/mask 运行时预处理。

抄 serve_my_chain + my_full_forward 的预处理段 + GR00T processor 源码(_apply_vlm_processing/collator/formalize)
+ transformers get_rope_index(纯函数)。绕过 Gr00tPolicy，只用 Qwen3VLProcessor(transformers) +
embed_tokens.npy 查表 + get_rope_index 纯函数 + torch mask。

板端不加载 backbone 2B。语言塔 input_embeds 由 embed_tokens 查表得到，position_ids/mask 本地算，
然后喂 language.hbm。
"""
import re
import cv2
import numpy as np
import torch
from transformers import AutoProcessor

IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
VISION_START_TOKEN_ID = 151652
SPATIAL_MERGE_SIZE = 2
MASK_NEG = -32767.0  # 对齐专家 leap_llm Spirit neg_mask_value / qwen3_vl mask_value
MAX_LEN = 256


def get_rope_index(input_ids, image_grid_thw, attention_mask):
    """抄 Qwen3VLModel.get_rope_index(transformers) 纯函数版，去 video 分支(GR00T 无 video)、去 self。
    只读 4 个 config 常量(spatial_merge_size/image_token_id/video_token_id/vision_start_token_id)。
    返回 position_ids [3, batch, seq] int。"""
    position_ids = torch.ones(3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device)
    image_index = 0
    for i in range(input_ids.shape[0]):
        ids = input_ids[i][attention_mask[i] == 1]
        vision_start_indices = torch.argwhere(ids == VISION_START_TOKEN_ID).squeeze(1)
        vision_tokens = ids[vision_start_indices + 1]
        image_nums = int((vision_tokens == IMAGE_TOKEN_ID).sum())
        input_tokens = ids.tolist()
        llm_pos_ids_list = []
        st = 0
        remain_images = image_nums
        for _ in range(image_nums):
            if IMAGE_TOKEN_ID in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(IMAGE_TOKEN_ID, st)
            else:
                ed_image = len(input_tokens) + 1
            t, h, w = image_grid_thw[image_index][0], image_grid_thw[image_index][1], image_grid_thw[image_index][2]
            image_index += 1
            remain_images -= 1
            ed = ed_image
            llm_grid_t = t.item()
            llm_grid_h = h.item() // SPATIAL_MERGE_SIZE
            llm_grid_w = w.item() // SPATIAL_MERGE_SIZE
            text_len = ed - st
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            llm_pos_ids_list.append(torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx)
            t_index = torch.arange(llm_grid_t, device=input_ids.device).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            h_index = torch.arange(llm_grid_h, device=input_ids.device).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w, device=input_ids.device).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w
        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx)
        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[:, i, attention_mask[i] == 1] = llm_positions
    return position_ids


def _smallest_max_size(img, m=256):
    H, W = img.shape[:2]
    s = m / float(min(H, W))
    return cv2.resize(img, (int(round(W * s)), int(round(H * s))), interpolation=cv2.INTER_AREA)


def resize_crop_256(img_uint8_hwc):
    """== Gr00t eval_image_transform == img_to_pixel 前半。产 [256,256,3] uint8。"""
    img = _smallest_max_size(img_uint8_hwc, 256)
    H, W = img.shape[:2]
    ch, cw = int(H * 0.95), int(W * 0.95)
    t, l = (H - ch) // 2, (W - cw) // 2
    img = img[t:t + ch, l:l + cw]
    img = _smallest_max_size(img, 256)
    H, W = img.shape[:2]
    t, l = max(0, (H - 256) // 2), max(0, (W - 256) // 2)
    return img[t:t + 256, l:l + 256]


class BoardPreprocess:
    def __init__(self, cosmos_dir, embed_npy, image_keys):
        self.proc = AutoProcessor.from_pretrained(cosmos_dir, trust_remote_code=True)
        self.proc.tokenizer.padding_side = "left"  # 抄 Gr00tN1d7DataCollator:169
        self.embed_table = np.load(embed_npy)  # [151936,2048] fp16
        self.image_keys = image_keys

    def __call__(self, text, images_uint8_hwc_dict):
        """text: str; images_uint8_hwc_dict: {view: [np.uint8 HWC]}。返回 language.hbm 喂入所需的全部张量。"""
        # formalize_language (抄 processing:678-680)
        lang = re.sub(r"[^\w\s]", "", text.lower())
        # resize/crop 256 → uint8 CHW tensor（传 Qwen3VLProcessor 算 grid_thw，pixel_values 板端丢弃）
        frames_t = []
        for view in self.image_keys:
            for img in images_uint8_hwc_dict[view]:
                a = resize_crop_256(np.asarray(img))
                frames_t.append(torch.from_numpy(a).permute(2, 0, 1))  # [3,256,256] uint8
        # _apply_vlm_processing (抄 :557-570) + collator tokenize (抄 :190-196)
        conversation = [{"role": "user", "content": [*[{"type": "image", "image": im} for im in frames_t], {"type": "text", "text": lang}]}]
        text_str = self.proc.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        vlm = self.proc(text=[text_str], images=list(frames_t), return_tensors="pt", padding="max_length", max_length=MAX_LEN)
        input_ids = vlm["input_ids"]
        attn_mask = vlm["attention_mask"]
        grid_thw = vlm["image_grid_thw"]
        # embed_tokens 查表（== my_language.embed_tokens，bf16→fp16）
        input_embeds = self.embed_table[input_ids.numpy()].astype(np.float16)  # [1,256,2048] fp16
        # get_rope_index → position_ids
        position_ids = get_rope_index(input_ids, grid_thw, attn_mask)
        # mask (抄 my_full_forward，专家双向：causal | pad q 行 | pad k 列)
        seq = input_ids.shape[1]
        keep = (attn_mask == 1)
        blocked = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
        blocked = blocked | ~keep.unsqueeze(1) | ~keep.unsqueeze(2)
        am = torch.zeros(seq, seq).masked_fill(blocked, float(MASK_NEG)).unsqueeze(1)  # [1,1,seq,seq]
        pad1d = (attn_mask == 0)
        pad_mask = torch.where(pad1d, float(MASK_NEG), 0.0).view(1, 1, 1, -1)  # [1,1,1,seq]
        img_pos = (input_ids == IMAGE_TOKEN_ID)
        return {
            "input_ids": input_ids.numpy(),                                  # [1,256] int
            "attn_mask": attn_mask.numpy(),                                  # [1,256] int
            "input_embeds": input_embeds,                                    # [1,256,2048] fp16
            "position_ids": position_ids.numpy().astype(np.int32),           # [3,1,256]
            "attention_mask": am.numpy().astype(np.float16),                 # [1,1,256,256]
            "pad_mask": pad_mask.numpy().astype(np.float16),                 # [1,1,1,256]
            "img_pos": img_pos.numpy(),                                      # [1,256] bool
            "grid_thw": grid_thw.numpy(),
        }
