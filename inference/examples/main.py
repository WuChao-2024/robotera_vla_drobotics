import rclpy
from rclpy.node import Node
import numpy as np
import time
from pynput import keyboard

# 假设你的硬件接口和模型封装
from hardware_interface import RobotInterface
from model_worker import OpenPIWorker
from config import OBS_STRUCTURE, MODEL_CFG, CONTROL_HZ
from stream import interpolate_actions
def contains_none(obj):
    if obj is None: return True
    if isinstance(obj, dict): return any(contains_none(v) for v in obj.values())
    if isinstance(obj, (list, tuple, set)): return any(contains_none(item) for item in obj)
    return False

class VlaSyncInferenceNode(Node):
    def __init__(self, init_vec):
        super().__init__('vla_sync_node')
        
        # 1. 硬件接口与模型初始化
        # 这里的 RobotInterface 建议使用独立的回调组防止阻塞 ROS2 通讯
        self.interface = RobotInterface(node_name='inference_node')
        self.worker = OpenPIWorker()
        self.init_vec = init_vec
        
        # 2. 状态变量
        self.need_reset = False
        self.is_running = True
        
        # 3. 参数配置
        self.CHUNK = MODEL_CFG.get('chunk_size', 16)
        self.EXECUTE_STEPS = MODEL_CFG.get('execute_steps', 80) # 通常只执行前 N 步
        self.ACTION_DT = 1.0 / CONTROL_HZ
        
        self.get_logger().info(f"Sync Inference Node Initialized. Target: {CONTROL_HZ}Hz")

    def run_loop(self):
        """主控制循环"""
        while rclpy.ok() and self.is_running:
            if self.need_reset:
                self.handle_system_reset()
                continue

            # --- STEP 1: 获取最新观测 (同步等待数据) ---
            obs = self.interface.get_observation()
            if contains_none(obs):
                self.get_logger().warn("Waiting for complete sensor data...", throttle_duration_sec=2.0)
                time.sleep(0.01)
                continue

            # 数据格式预处理 (对应你原本在进程里的逻辑)
            imgs_input = {
                'cam_high': np.transpose(obs['images']['cam_head'], (2, 0, 1)),
                'cam_left_wrist': np.transpose(obs['images']['cam_left'], (2, 0, 1)),
                'cam_right_wrist': np.transpose(obs['images']['cam_right'], (2, 0, 1))
            }
            state_input = np.concatenate([obs['state'][comp.name] for comp in OBS_STRUCTURE]).astype(np.float32)

            # --- STEP 2: 推理 (阻塞当前线程) ---
            t0 = time.time()
            try:
                # 同步推理不再需要复杂的 prev_actions 逻辑，除非你要闭环反馈
                new_actions = self.worker.infer(imgs_input, state_input)
                inference_time = time.time() - t0
                self.get_logger().info(f"Inference OK ({inference_time:.3f}s). Executing {self.EXECUTE_STEPS} steps.")
            except Exception as e:
                self.get_logger().error(f"Inference failed: {e}")
                continue

            # --- STEP 3: 执行 Action Chunk ---
            if new_actions is not None:
                interpolated_actions = interpolate_actions(new_actions, num_points=100)
                for i in range(self.EXECUTE_STEPS):
                    step_start = time.time()
                    
                    if self.need_reset or not self.is_running:
                        break
                    
                    # 发布动作
                    # print(new_actions[i])
                    self.interface.publish_action(interpolated_actions[i])
                    
                    # 严格控制控制频率 (例如 50Hz)
                    elapsed = time.time() - step_start
                    if elapsed < self.ACTION_DT:
                        time.sleep(self.ACTION_DT - elapsed)
            
            # 执行完一轮后直接进入下一次循环（即：获取新观测 -> 再次推理）

    def handle_system_reset(self):
        self.get_logger().info("RESET START")
        self.interface.smooth_reset(self.init_vec, duration=2.0)
        self.need_reset = False
        self.get_logger().info("RESET DONE")

def main():
    rclpy.init()
    
    init_vec = np.array([
        0.2, -0.35, 0.2, 0.5, 0.5, 0.5, 0.5,
        1.46, -0.39, 0.001, 0.00, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26,
        0.2, 0.35, 0.2, -0.5, 0.5, -0.5, 0.5,
        1.46, -0.39, 0.001, 0.00, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26,
    ])

    node = VlaSyncInferenceNode(init_vec)

    # 键盘监听 (保持非阻塞)
    def on_press(key):
        try:
            if key.char == 'r': node.need_reset = True
            if key.char == 'q': 
                node.is_running = False
                rclpy.shutdown()
        except: pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # 启动循环
    try:
        # 使用自定义循环代替简单的 spin，方便处理阻塞推理后的动作序列执行
        node.run_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.interface.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()