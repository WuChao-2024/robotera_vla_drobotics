import numpy as np
from scipy.spatial.transform import Rotation, RotationSpline
from scipy.interpolate import CubicSpline

def interpolate_actions(actions, num_points=100):
    """
    将动作序列从20个点插值到100个点
    
    Args:
        actions: shape (20, 38) 的动作数组
        num_points: 目标点数 (默认100)
    
    Returns:
        interpolated_actions: shape (100, 38) 的插值结果
    """
    n_original = len(actions)
    n_target = num_points
    
    # 创建插值参数（时间点）
    t_original = np.arange(n_original)
    t_target = np.linspace(0, n_original - 1, n_target)
    
    # 初始化结果数组
    interpolated = np.zeros((n_target, actions.shape[1]))
    
    # 分离各组件
    right_end_pose = actions[:, 0:7]        # [x, y, z, qx, qy, qz, qw]
    right_hand_qpos = actions[:, 7:19]      # 12个关节位置
    left_end_pose = actions[:, 19:26]       # [x, y, z, qx, qy, qz, qw]
    left_hand_qpos = actions[:, 26:38]      # 12个关节位置
    
    # 1. 插值位置（xyz）使用三次样条
    for i in range(3):
        cs = CubicSpline(t_original, right_end_pose[:, i], bc_type='natural')
        interpolated[:, i] = cs(t_target)
        cs = CubicSpline(t_original, left_end_pose[:, i], bc_type='natural')
        interpolated[:, i+19] = cs(t_target)
    
    # 2. 插值四元数使用 RotationSpline
    # 创建 Rotation 对象
    right_rots = Rotation.from_quat(right_end_pose[:, 3:7])  # [qx, qy, qz, qw]
    left_rots = Rotation.from_quat(left_end_pose[:, 3:7])
    
    # 创建旋转样条插值器
    right_spline = RotationSpline(t_original, right_rots)
    left_spline = RotationSpline(t_original, left_rots)
    
    # 在目标时间点进行插值
    right_interp = right_spline(t_target)
    left_interp = left_spline(t_target)
    
    # 将插值结果写入数组
    interpolated[:, 3:7] = right_interp.as_quat()
    interpolated[:, 22:26] = left_interp.as_quat()
    
    # 3. 插值关节位置使用三次样条
    for i in range(12):
        # 右手关节
        cs = CubicSpline(t_original, right_hand_qpos[:, i], bc_type='natural')
        interpolated[:, 7+i] = cs(t_target)
        # 左手关节
        cs = CubicSpline(t_original, left_hand_qpos[:, i], bc_type='natural')
        interpolated[:, 26+i] = cs(t_target)
    
    return interpolated