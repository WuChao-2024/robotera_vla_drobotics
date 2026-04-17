import dataclasses
from typing import ClassVar

import einops
import numpy as np
import cv2
from training.interfaces import transforms
from scipy.spatial.transform import Rotation as R


def make_aloha_example() -> dict:
    """Creates a random input example for the Aloha policy."""
    return {
        # "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            # "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }

@dataclasses.dataclass(frozen=True)
class M7Inputs(transforms.DataTransformFn):
    """Inputs for the Aloha policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [14]  # 末端空间状态 [16] 
    - actions: [action_horizon, 14]  # 末端空间动作 [action_horizon, 16]
    """

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    # adapt_to_pi: bool = True

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        data = _decode_input(data)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that base image always exists.
        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add the extra images.
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": np.asarray(data["state"]),
            # "raw_state": np.asarray(data["state"]),
            # "raw_state": np.concatenate([
            #     data["state"][..., 21:28],   # left_arm_end
            #     data["state"][..., 40:52],   # left_hand
            #     data["state"][..., 14:21],   # right_arm_end
            #     data["state"][..., 28:40],   # right_hand
            # ], axis=0)
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            # actions = np.hstack([actions[..., 19:], actions[..., :19]])
            inputs["actions"] = actions


        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs

class M7Outputs(transforms.DataTransformFn):
    """Outputs for the Aloha policy."""

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.

    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims.
        actions = np.asarray(data["actions"][:, :])
        return {"actions": actions}

def _decode_input(data: dict) -> dict:
    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        img = einops.rearrange(img, "c h w -> h w c")
        
        # Pad and resize to 640x480
        img = pad_and_resize_to_target(img, target_height=480, target_width=640)
        return img
    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict

    data["state"] = np.asarray(data["state"])

    return data

def pad_and_resize_to_target(image, target_height=480, target_width=640):
    """
    Pad image to square then resize to target dimensions
    
    Args:
        image: numpy array of shape [H, W, C]
        target_height: desired height
        target_width: desired width
    
    Returns:
        padded and resized image of shape [target_height, target_width, C]
    """
    h, w = image.shape[:2]
    
    # Calculate padding to make image square
    if h > w:
        # Pad width
        pad_total = h - w
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        pad_top = 0
        pad_bottom = 0
    else:
        # Pad height
        pad_total = w - h
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        pad_left = 0
        pad_right = 0
    
    # Apply padding
    padded = cv2.copyMakeBorder(
        image, 
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, 
        value=[0, 0, 0]  # Black padding
    )
    
    # Resize to target dimensions
    resized = cv2.resize(padded, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    
    return resized
