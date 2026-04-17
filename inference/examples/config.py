import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class Component:
    name: str
    dim: int

# ==================== 1. 机器人硬件关节定义 ====================
JOINT_NAMES = {
    'right_arm': [
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_arm_yaw_joint",
        "right_elbow_pitch_joint", "right_elbow_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint"
    ],
    'left_arm': [
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_arm_yaw_joint",
        "left_elbow_pitch_joint", "left_elbow_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint"
    ],
    'right_hand': [
        "right_hand_thumb_bend_joint", "right_hand_thumb_rota_joint1", "right_hand_thumb_rota_joint2",
        "right_hand_index_bend_joint", "right_hand_index_joint1", "right_hand_index_joint2",
        "right_hand_mid_joint1", "right_hand_mid_joint2", "right_hand_ring_joint1",
        "right_hand_ring_joint2", "right_hand_pinky_joint1", "right_hand_pinky_joint2"
    ],
    'left_hand': [
        "left_hand_thumb_bend_joint", "left_hand_thumb_rota_joint1", "left_hand_thumb_rota_joint2",
        "left_hand_index_bend_joint", "left_hand_index_joint1", "left_hand_index_joint2",
        "left_hand_mid_joint1", "left_hand_mid_joint2", "left_hand_ring_joint1", 
        "left_hand_ring_joint2", "left_hand_pinky_joint1", "left_hand_pinky_joint2"
    ],
    'waist': ['waist_roll_joint', 'waist_pitch_joint', 'waist_yaw_joint'],
    'neck': ['neck_yaw_joint', 'neck_pitch_joint']
}

# ==================== 2. 语义索引自动生成 (单一事实来源) ====================
# 严格对齐训练数据的观测拼接顺序
OBS_STRUCTURE = [
    Component('right_arm_qpos', 7),
    Component('left_arm_qpos', 7),
    Component('right_end_pose', 7),
    Component('left_end_pose', 7),
    Component('right_hand_qpos', 12),
    Component('left_hand_qpos', 12),
    Component('waist_qpos', 3),
    Component('neck_qpos', 2),
]

# 严格对齐模型生成的动作输出结构
ACT_STRUCTURE = [
    Component('right_end_pose', 7),
    Component('right_hand_qpos', 12),
    Component('left_end_pose', 7),
    Component('left_hand_qpos', 12),
]

def generate_indexer(structure: List[Component]):
    indexer = {}
    curr = 0
    for c in structure:
        indexer[c.name] = slice(curr, curr + c.dim)
        curr += c.dim
    return indexer, curr

STATE_INDEXER, STATE_DIM = generate_indexer(OBS_STRUCTURE)
ACTION_INDEXER, ACTION_DIM = generate_indexer(ACT_STRUCTURE)

# ==================== 3. 硬件与模型超参数 ====================
MODEL_CFG = {
    "checkpoint_path": "/workspace/algorithm/robotera_vla/checkpoints/pi05_M7_pp_opensource/260322/100000",
    "policy_name": "pi05_M7_pp_opensource",
    "prompt": "pick the apple and put it in the bowl.",
    "chunk_size": 20,
    "action_fps": 20,
}
CONTROL_HZ = 100