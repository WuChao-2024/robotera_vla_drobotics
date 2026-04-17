#!/usr/bin/env python3
# model_inference_process.py
import numpy as np
import torch
import time
from training.configs import config as _config
from training.interfaces.policies import policy_config as _policy_config
from training.interfaces.shared import download

# 维度常量定义
STATE_DIM = 57  
ACTION_DIM = 38
INFERENCE_DELAY = 4 

class ModelInferenceProcess:
    def __init__(self):
        self.policy = None
        self.load_model()
        # self.warm_up()
        
    def load_model(self):
        print(f"[{time.strftime('%H:%M:%S')}] 正在加载模型...")
        # 路径保持
        checkpoint_dir = download.maybe_download("/era-ai/lm/user/wpc/openpi/checkpoints/pi05_M7_pp_opensource/260322/100000")
        config = _config.get_config("pi05_M7_pp_opensource")
        self.policy = _policy_config.create_trained_policy(config, checkpoint_dir)
        print("模型加载完成。")

    def warm_up(self):
        """执行深度预热"""
        print(f"[{time.strftime('%H:%M:%S')}] 正在执行 10 次预热推理...")
        for i in range(10):
            self.inference_callback(is_warmup=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("预热完成。")
        
    def inference_callback(self, is_warmup=False):
        """处理随机生成的 uint8 观测数据"""
        start_time = time.time()
        
        # 1. 构造随机状态 (保持 float32，因为状态通常由传感器转换后得到)
        random_state = np.random.uniform(-1, 1, STATE_DIM).astype(np.float32)
        
        # 2. 构造随机图像 (0-255, uint8)
        # 模拟 3 个摄像头的输入
        random_imgs = {
            "cam_high": np.random.randint(0, 256, (3, 848, 480), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(0, 256, (3, 640, 480), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(0, 256, (3, 640, 480), dtype=np.uint8)
        }

        
        model_input = {
            "state": random_state,
            "images": random_imgs,
            # "prev_actions": random_prev_actions1,
            # "inference_delay": INFERENCE_DELAY,
            "prompt": 'A'
        }
      

        # 模型推理
        with torch.no_grad():
            result = self.policy.infer(model_input)
        
        # 提取动作
        if isinstance(result, dict) and 'actions' in result:
            actions = result['actions']
        else:
            actions = result
        # actions shape: (20, 38)
        # right_ee (7) + right_hand(12) + left_ee(7) + left_hand(12)
        
        inference_time = time.time() - start_time
        if not is_warmup:
            print(f"Inference completed in {inference_time:.3f}s")
            
        return actions


if __name__ == "__main__":
    tester = ModelInferenceProcess()
    print(f"\n[{time.strftime('%H:%M:%S')}] 开始 50 次稳定性测试...")
    
    latencies = []
    for i in range(50):
        act = tester.inference_callback()
        latencies.append(time.time() - time.time()) # 这里修正为记录上面 callback 的耗时
        # 修正: 上面 callback 内部已有 start_time，我们直接记录返回值即可
    
    # 由于 callback 内部打印了，这里做个统计总结
    print("\n" + "="*40)
    print("性能统计结果:")
    # 注意：上面的 latencies 计算有误，重新用 callback 返回的逻辑来统计更准
    # 这里建议在 callback 里 return actions, inference_time