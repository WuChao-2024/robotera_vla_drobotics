import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage, JointState
from geometry_msgs.msg import PoseStamped
from xbot_common_interfaces.msg import ServoPose, HybridJointCommand
import time
import numpy as np
import threading
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from frame_decoder import H264FrameDecoder
from config import JOINT_NAMES, ACTION_INDEXER

class RobotInterface:
    def __init__(self, node_name="robot_interface"):
        if not rclpy.ok(): rclpy.init()
        self.node = Node(node_name)
        self.cb_group = ReentrantCallbackGroup()
        
        self.decoders = {
            'cam_head': H264FrameDecoder(),
            'cam_left': H264FrameDecoder(),
            'cam_right': H264FrameDecoder()
        }

        self.imgs = {'cam_head': None, 'cam_left': None, 'cam_right': None}
        # 初始化所有状态字段
        self.states = {
            'left_hand_qpos': np.zeros(12),
            'right_hand_qpos': np.zeros(12),
            'left_arm_qpos': np.zeros(7),
            'right_arm_qpos': np.zeros(7),
            'left_end_pose': np.zeros(7),
            'right_end_pose': np.zeros(7),
            'neck_qpos': np.zeros(2),
            'waist_qpos': np.zeros(3)
        }

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # 订阅
        if node_name == "inference_node":
            self.node.create_subscription(CompressedImage, "/teleop/camera_high_image_h264", self._img_head_cb, qos)
            self.node.create_subscription(CompressedImage, "/teleop/camera_left_image_h264", self._img_left_cb, qos)
            self.node.create_subscription(CompressedImage, "/teleop/camera_right_image_h264", self._img_right_cb, qos)
        self.node.create_subscription(JointState, '/joint_states', self._joint_state_cb, qos, callback_group=self.cb_group)
        self.node.create_subscription(ServoPose, '/get_pose', self._end_state_cb, qos, callback_group=self.cb_group)

        # 发布
        self.pose_pub = self.node.create_publisher(ServoPose, '/servo_poses', 10)
        self.hand_pub = self.node.create_publisher(HybridJointCommand, '/hand_controller/commands', 10)

        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()

    def _img_head_cb(self, msg):
        # print('--------')
        img = self.decoders['cam_head'].decode_one_frame(bytes(msg.data))
        # print(img)
        if img is not None: self.imgs['cam_head'] = img[:, :, ::-1]

    def _img_left_cb(self, msg):
        img = self.decoders['cam_left'].decode_one_frame(bytes(msg.data))
        if img is not None: self.imgs['cam_left'] = img[:, :, ::-1]

    def _img_right_cb(self, msg):
        img = self.decoders['cam_right'].decode_one_frame(bytes(msg.data))
        if img is not None: self.imgs['cam_right'] = img[:, :, ::-1]

    def _end_state_cb(self, msg: ServoPose):
        def p2a(ps):
            p, o = ps.pose.position, ps.pose.orientation
            return np.array([p.x, p.y, p.z, o.x, o.y, o.z, o.w], dtype=np.float32)
        self.states['left_end_pose'] = p2a(msg.left_pose)
        self.states['right_end_pose'] = p2a(msg.right_pose)

    def _joint_state_cb(self, msg: JointState):
        try:
            name_to_pos = {n: p for n, p in zip(msg.name, msg.position)}
            for key, names in JOINT_NAMES.items():
                if f"{key}_qpos" in self.states:
                    self.states[f"{key}_qpos"] = np.array([name_to_pos[n] for n in names if n in name_to_pos], dtype=np.float32)
        except Exception: pass

    def get_observation(self):
        return {'images': self.imgs.copy(), 'state': self.states.copy(), 't': time.time()}

    def publish_action(self, action_vec):
        """通过语义索引解析动作并发布"""
        p_msg = ServoPose()
        for side in ['left', 'right']:
            data = action_vec[ACTION_INDEXER[f'{side}_end_pose']]
            ps = PoseStamped()
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, data[:3])
            ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = map(float, data[3:])
            setattr(p_msg, f'{side}_pose', ps)
        self.pose_pub.publish(p_msg)

        h_msg = HybridJointCommand()
        h_msg.joint_name = [
            "left_hand_thumb_bend_joint",
            "left_hand_thumb_rota_joint1",
            "left_hand_thumb_rota_joint2",
            "left_hand_index_bend_joint",
            "left_hand_index_joint1",
            "left_hand_index_joint2",
            "left_hand_mid_joint1",
            "left_hand_mid_joint2",
            "left_hand_ring_joint1",
            "left_hand_ring_joint2",
            "left_hand_pinky_joint1",
            "left_hand_pinky_joint2",
            "right_hand_thumb_bend_joint",
            "right_hand_thumb_rota_joint1",
            "right_hand_thumb_rota_joint2",
            "right_hand_index_bend_joint",
            "right_hand_index_joint1",
            "right_hand_index_joint2",
            "right_hand_mid_joint1",
            "right_hand_mid_joint2",
            "right_hand_ring_joint1",
            "right_hand_ring_joint2",
            "right_hand_pinky_joint1",
            "right_hand_pinky_joint2",
        ]
        h_msg.velocity = [0.0] * len(h_msg.joint_name)
        h_msg.kp = [100.0] * len(h_msg.joint_name)
        h_msg.kd = [0.0] * len(h_msg.joint_name)
        h_msg.feedforward = [350.0] * len(h_msg.joint_name)
        h_msg.position = action_vec[ACTION_INDEXER['left_hand_qpos']].tolist() + \
                         action_vec[ACTION_INDEXER['right_hand_qpos']].tolist()
        self.hand_pub.publish(h_msg)
        # print('0000')

    def smooth_reset(self, target_vec, duration=2.0, hz=50):

        curr_obs = self.get_observation()
        curr_state = curr_obs['state']
        
        steps = int(duration * hz)
        fractions = np.linspace(0, 1, steps)
        
        # 准备旋转插值器 (针对四元数部分)
        slerps = {}
        for side in ['left', 'right']:
            idx = ACTION_INDEXER[f'{side}_end_pose']
            # 提取当前和目标的四元数 [x, y, z, w]
            q_start = curr_state[f'{side}_end_pose'][3:7]
            q_target = target_vec[idx][3:7]
            
            rots = R.from_quat([q_start, q_target])
            slerps[side] = Slerp([0, 1], rots)

        print(f"[Hardware] Starting reset sequence ({duration}s)...")

        for f in fractions:
            # 使用平滑余弦曲线优化进度 (可选，让动作两头慢中间快)
            f_smooth = 0.5 * (1 - np.cos(f * np.pi))
            
            # 构造临时的动作向量
            interp_vec = target_vec.copy()
            
            for side in ['left', 'right']:
                idx = ACTION_INDEXER[f'{side}_end_pose']
                # h_idx = ACTION_INDEXER[f'{side}_hand_qpos']
                
                # 1. 位置线性插值
                interp_vec[idx][:3] = (1 - f_smooth) * curr_state[f'{side}_end_pose'][:3] + \
                                    f_smooth * target_vec[idx][:3]
                
                # 2. 旋转 Slerp 插值
                interp_vec[idx][3:7] = slerps[side](f_smooth).as_quat()
                
                # 3. 手部关节线性插值
                # interp_vec[h_idx] = (1 - f_smooth) * curr_state[f'{side}_hand_qpos'] + \
                #                     f_smooth * target_vec[h_idx]
            
            # 执行发布
            self.publish_action(interp_vec)
            
            # 此处的 sleep 仅阻塞主线程，不影响订阅回调更新 self.states
            time.sleep(1.0 / hz)

        self.publish_action(target_vec)
        print("[Hardware] Reset complete.")

    def stop(self):
        self.node.destroy_node()
        rclpy.shutdown()