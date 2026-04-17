import time
import torch
import numpy as np
from training.configs import config as _config
from training.interfaces.policies import policy_config as _policy_config
from training.interfaces.shared import download
from config import MODEL_CFG, ACTION_DIM, STATE_DIM

class OpenPIWorker:
    def __init__(self):
        self.policy = None
        self.load_and_warmup()

    def load_and_warmup(self):
        """加载并针对 'None' 和 'Fixed Length' 两种输入形状进行预热"""
        print(f"[Model] Loading: {MODEL_CFG['checkpoint_path']}")
        try:
            checkpoint_dir = download.maybe_download(MODEL_CFG['checkpoint_path'])
            config = _config.get_config(MODEL_CFG['policy_name'])
            self.policy = _policy_config.create_trained_policy(config, checkpoint_dir)
            print("[Model] Policy loaded.")
            
            # 预热数据模版
            dummy_imgs = {k: np.random.randint(0, 256, (3, 224, 224), dtype=np.uint8) for k in ["cam_high", "cam_left_wrist", "cam_right_wrist"]}
            dummy_state = np.random.randn(STATE_DIM).astype(np.float32)
            
            # --- 模式 A: 无前缀预热 (针对第一帧) ---
            print("[Model] Warmup Phase 1: No prefix...")
            input_no_prefix = {
                "images": dummy_imgs,
                "state": dummy_state,
                "prompt": MODEL_CFG['prompt'],
                # "prev_actions": None  # 关键点
            }
            for _ in range(3):
                with torch.no_grad():
                    _ = self.policy.infer(input_no_prefix)
            
        except Exception as e:
            raise RuntimeError(f"Model initialization failed: {e}")

    def infer(self, imgs, state):
        """执行推理，prev_actions 为 None 或 长度为 L 的 ndarray"""
        inputs = {
            "images": imgs,
            "state": state,
            "prompt": MODEL_CFG['prompt'],
            # "prev_actions": prev_actions,
        }
            
        with torch.no_grad():
            result = self.policy.infer(inputs)
        return result['actions']