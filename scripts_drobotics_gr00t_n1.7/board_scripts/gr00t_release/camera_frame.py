"""pi05 相机系 SE(3) EE-delta 转化（移植自 robotera_vla_drobotics，纯 numpy 版）。

来源:robotera_vla_drobotics/training/interfaces/transforms.py 的
CamDeltaEeActions(272-335)/ CamAbsoluteEeActions(374-439)。

fk_base_rs 用纯 numpy 复刻 pinocchio FK（base_link→stereo_link 5 关节链：
waist yaw/roll/pitch + neck yaw/pitch）。常量从 l3_4.urdf 提取，本机对 pinocchio
验证 max diff 2.2e-16（2000 随机关节角，机器精度）。去 pinocchio 依赖（板端 aarch64
无 bindings wheel），serve 自包含 numpy+scipy。

末端位姿 7 维 [x,y,z,qx,qy,qz,qw](xyzw)，与 pi05/M7 数据集一致。
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rotation

# base_link→stereo_link 链：5 关节 (jid, 旋转轴, qpos 索引)
#   waist_yaw(13,RZ,q12) waist_roll(14,RX,q13) waist_pitch(15,RY,q14)
#   neck_yaw(35,RZ,q34) neck_pitch(36,RY,q35)
# qpos 映射(复刻 pi05 get_rs_frame): q12=waist[2], q13=waist[0], q14=waist[1], q34=neck[0], q35=neck[1]
_CHAIN = [(13, 'z', 12), (14, 'x', 13), (15, 'y', 14), (35, 'z', 34), (36, 'y', 35)]
# 各关节 origin（parent joint frame→this joint frame 的固定变换，从 URDF 提取）
_ORIGINS = {
    13: np.array([[1.0, 0.0, 0.0, -0.04724], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0707], [0.0, 0.0, 0.0, 1.0]]),
    14: np.eye(4),
    15: np.eye(4),
    35: np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.398], [0.0, 0.0, 0.0, 1.0]]),
    36: np.array([[1.0, 0.0, 0.0, 0.04], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0905], [0.0, 0.0, 0.0, 1.0]]),
}
# stereo_link 在 neck_pitch(36) frame 的固定外参
_STEREO_PF = np.array([
    [0.8660247915829389, 0.0, 0.5000010603626028, 0.059712],
    [0.0, 1.0, 0.0, 0.0],
    [-0.5000010603626028, 0.0, 0.8660247915829389, 0.13282],
    [0.0, 0.0, 0.0, 1.0],
])


def _RAXIS(axis, q):
    c, s = np.cos(q), np.sin(q)
    if axis == 'x':
        return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])
    if axis == 'y':
        return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1.0]])
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def fk_base_rs(waist_rpy, neck_yp):
    """waist(3) + neck(2) 关节角 -> stereo_link 相对 base_link 的 4x4 外参 T_base_rs。

    纯 numpy 复刻 pinocchio FK（本机验证 max diff 2.2e-16）。
    qpos 重排约定（pi05 get_rs_frame）: q[12]=waist[2], q[13]=waist[0], q[14]=waist[1], q[34:36]=neck。
    """
    waist_rpy = np.asarray(waist_rpy, dtype=float).reshape(-1)
    neck_yp = np.asarray(neck_yp, dtype=float).reshape(-1)
    q = {12: waist_rpy[2], 13: waist_rpy[0], 14: waist_rpy[1], 34: neck_yp[0], 35: neck_yp[1]}
    T = np.eye(4)
    for _jid, axis, qi in _CHAIN:
        T = T @ _ORIGINS[_jid] @ _RAXIS(axis, q[qi])
    return T @ _STEREO_PF


def _quatpos_to_H(qp):
    qp = np.asarray(qp, dtype=float)
    if qp.ndim == 1:
        qp = qp[None, :]
    n = qp.shape[0]
    H = np.zeros((n, 4, 4))
    H[:, 3, 3] = 1.0
    H[:, :3, 3] = qp[:, :3]
    H[:, :3, :3] = Rotation.from_quat(qp[:, 3:7]).as_matrix()
    return H


def _H_to_quatpos(H):
    H = np.asarray(H, dtype=float)
    if H.ndim == 2:
        H = H[None, :]
    n = H.shape[0]
    qp = np.zeros((n, 7))
    qp[:, :3] = H[:, :3, 3]
    qp[:, 3:] = Rotation.from_matrix(H[:, :3, :3]).as_quat()
    return qp


def cam_relative(action_tx7, anchor_state_7, t_base_rs):
    action_tx7 = np.asarray(action_tx7, dtype=float)
    if action_tx7.ndim == 1:
        action_tx7 = action_tx7[None, :]
    anchor_state_7 = np.asarray(anchor_state_7, dtype=float).reshape(-1)
    inv_t = np.linalg.inv(t_base_rs)
    t_act_cam = inv_t @ _quatpos_to_H(action_tx7)
    act_cam_qp = _H_to_quatpos(t_act_cam)
    t_state_cam = (inv_t @ _quatpos_to_H(anchor_state_7))[0]
    state_cam_qp = _H_to_quatpos(t_state_cam[None])[0]
    delta_t = act_cam_qp[:, :3] - state_cam_qp[:3]
    r_state_cam = Rotation.from_quat(state_cam_qp[3:7]).as_matrix()
    r_act_cam = Rotation.from_quat(act_cam_qp[:, 3:7]).as_matrix()
    delta_r = np.linalg.inv(r_state_cam) @ r_act_cam
    delta_q = Rotation.from_matrix(delta_r).as_quat()
    return np.concatenate([delta_t, delta_q], axis=1)


def cam_absolute(delta_tx7, anchor_state_7, t_base_rs):
    """单 EE key: 相机系 delta (T,7) + anchor state (7,) + T_base_rs (4,4) -> base 绝对 (T,7)。"""
    delta_tx7 = np.asarray(delta_tx7, dtype=float)
    if delta_tx7.ndim == 1:
        delta_tx7 = delta_tx7[None, :]
    anchor_state_7 = np.asarray(anchor_state_7, dtype=float).reshape(-1)
    inv_t = np.linalg.inv(t_base_rs)
    t_state_cam = (inv_t @ _quatpos_to_H(anchor_state_7))[0]
    state_cam_qp = _H_to_quatpos(t_state_cam[None])[0]
    abs_cam_t = delta_tx7[:, :3] + state_cam_qp[:3]
    r_state_cam = Rotation.from_quat(state_cam_qp[3:7]).as_matrix()
    r_delta = Rotation.from_quat(delta_tx7[:, 3:7]).as_matrix()
    abs_cam_r = r_state_cam @ r_delta
    abs_cam_h = np.zeros((delta_tx7.shape[0], 4, 4))
    abs_cam_h[:, 3, 3] = 1.0
    abs_cam_h[:, :3, 3] = abs_cam_t
    abs_cam_h[:, :3, :3] = abs_cam_r
    t_base_arm = t_base_rs @ abs_cam_h
    return _H_to_quatpos(t_base_arm)


if __name__ == "__main__":
    # ponytail 自检：fk_base_rs numpy vs 几组已知关节角输出形状 + cam_relative/cam_absolute 互逆
    w = np.array([0.1, -0.2, 0.3]); n = np.array([0.05, -0.15])
    T = fk_base_rs(w, n)
    assert T.shape == (4, 4) and abs(np.linalg.det(T[:3, :3]) - 1) < 1e-9
    st = np.array([0.1, 0.2, 0.3, *Rotation.from_euler('xyz', [0.1, 0.2, 0.3]).as_quat()])
    act = np.array([[0.4, 0.1, 0.5, *Rotation.from_euler('xyz', [0.5, -0.3, 0.2]).as_quat()]])
    rel = cam_relative(act, st, T)
    back = cam_absolute(rel, st, T)
    print("cam_relative->cam_absolute roundtrip max err:", np.abs(back - act).max())
    assert np.abs(back - act).max() < 1e-9
    print("camera_frame numpy self-check OK")
