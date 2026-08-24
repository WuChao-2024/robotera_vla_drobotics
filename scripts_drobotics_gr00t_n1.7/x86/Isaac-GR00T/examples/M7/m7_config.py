# M7 dual-arm humanoid modality config.
# All action groups use NON_EEF: every dimension (incl. quaternion components in
# end_pose) is treated as an independent scalar and min-max normalized. This matches
# the official SO-100 joint-space convention and keeps the dataset byte-for-byte
# unchanged. end_pose uses RELATIVE (delta from current state), hands use ABSOLUTE.

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

# Cosmos-Reason2-2B 是 gated repo, build_processor 联网会 401; 本地有完整副本,
# patch build_processor 把 repo id 重定向到本地 (GR00T_COSMOS_PATH), 覆盖 collator 等
# 所有 build_processor(model_name) 调用, 不动 GR00T 源码。
import os as _os
try:
    from gr00t.model.gr00t_n1d7 import processing_gr00t_n1d7 as _pg
    _orig_build_processor = _pg.build_processor

    def _build_processor_local(model_name, *a, **kw):
        if model_name == "nvidia/Cosmos-Reason2-2B":
            model_name = _os.environ.get("GR00T_COSMOS_PATH", model_name)
        return _orig_build_processor(model_name, *a, **kw)

    _pg.build_processor = _build_processor_local

    # M7 三相机异分辨率(cam_high 848x480, cam_left/right 640x480): transform 后
    # 多视角尺寸不同 stack 崩。patch _get_vlm_inputs, transform 后 center-crop
    # 所有视角到 256x256 方形(保比例, 只裁宽), 再 stack。
    import torch as _torch
    import torch.nn.functional as _F

    def _get_vlm_inputs_uniform(self, image_keys, images, masks, image_transform, language):
        temporal_stacked_images = {}
        if self.use_albumentations:
            replay = None
            for view in image_keys:
                view_masks = masks.get(view) if masks else None
                ti, replay = _pg.apply_with_replay(
                    image_transform, images[view], view_masks, replay
                )
                temporal_stacked_images[view] = _torch.stack(ti)
        else:
            for view in image_keys:
                temporal_stacked_images[view] = _torch.stack(
                    [image_transform(img) for img in images[view]]
                )
        tH, tW = 256, 256
        for k, v in temporal_stacked_images.items():
            H, W = v.shape[-2:]
            top = max(0, (H - tH) // 2)
            left = max(0, (W - tW) // 2)
            v = v[..., top:top + tH, left:left + tW]
            if v.shape[-2:] != (tH, tW):
                v = _F.interpolate(
                    v.float(), size=(tH, tW), mode="bilinear", align_corners=False
                ).to(_torch.uint8)
            temporal_stacked_images[k] = v
        stacked = _torch.stack(
            [temporal_stacked_images[v] for v in image_keys], dim=1
        ).flatten(0, 1)
        return self._apply_vlm_processing(stacked, language)

    _pg.Gr00tN1d7Processor._get_vlm_inputs = _get_vlm_inputs_uniform
except Exception as _e:
    print(f"[m7_config] build_processor patch skipped: {_e}")


m7_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam_high", "cam_left", "cam_right"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "right_arm_joint",
            "left_arm_joint",
            "right_end_pose",
            "left_end_pose",
            "right_hand",
            "left_hand",
            "waist",
            "neck",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["right_end_pose", "right_hand", "left_end_pose", "left_hand"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                camera_frame=True,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                camera_frame=True,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(m7_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
