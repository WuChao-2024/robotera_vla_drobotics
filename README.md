# Robotera VLA

[![OS](https://img.shields.io/badge/OS-Ubuntu%2022.04-orange.svg)](https://ubuntu.com/)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](https://www.docker.com/)
[![GPU](https://img.shields.io/badge/NVIDIA-CUDA%2012.4-green.svg)](https://developer.nvidia.com/cuda-zone)

Robotera VLA provides the `release_1.0` baseline for Robotera M7 data collection, training, and inference workflows. The repository focuses on the public-facing pieces around that stack: collection instructions, dataset and interface contracts, training baseline code, and an inference deployment example.

The repository assumes existing Robotera teleoperation and recorder software on the robot side. Those runtime services are not implemented here.

## M7 Baseline

This project currently targets Robotera M7 as the default robot baseline.

![M7 robot overview](docs/assets/m7_manual/image1.jpg)

| Item | Value |
|---|---|
| Model | M7 |
| DOF | 43 |
| Battery | 57.6V 15Ah 864Wh |
| Compute | 80 TOPS (x86) + 275 TOPS (Orin AGX) |
| Interfaces | Ethernet / USB 4.0 / Wi-Fi 6 |
| ROS2 Baseline | Humble, `ROS_DOMAIN_ID=211`, `rmw_cyclonedds_cpp` |

Detailed hardware and runtime constraints are documented in `docs/HARDWARE_SOFTWARE_REQUIREMENTS.md`.

## Repository Layout

- `data_collection/`: offline acquisition instructions, operation manuals, dataset schema, and lightweight validation examples
- `training/`: training baseline, configs, model wrappers, and example fine-tuning workflow
- `inference/`: inference deployment example, ROS 2 integration code, and interface definitions
- `docs/`: retained project-level hardware and software baseline

## Start Here

1. Review `docs/HARDWARE_SOFTWARE_REQUIREMENTS.md`.
2. Follow `data_collection/README.md` for offline data acquisition and export.
3. Review `data_collection/interfaces/dataset_schema.md` for dataset layout and field requirements.
4. Review `training/README.md` for the current training baseline and example checkpoint references.
5. Review `inference/README.md` for inference deployment and ROS 2 integration.
6. Review `inference/interfaces/ros2_api_contract.md` for observation and action message contracts.

## Current Surface

- Data collection is documented around the existing Robotera XOS and Meta Quest teleoperation workflow.
- Training includes the current M7-oriented baseline and related configs.
- Inference includes a Docker-based example and Robotera interface definitions for integration on an inference PC.

## PyTorch 推理支持

本项目基于 [OpenPI](https://github.com/Open-Pi/OpenPI) 的 PyTorch 模型实现，新增了 JAX 权重转 PyTorch `.safetensors` 的能力，以及纯 PyTorch 的推理路径。

### 目录结构

```
training/models_pytorch/
├── __init__.py
├── pi0_pytorch.py                  # PI0/PI05 PyTorch 模型（推理用）
├── gemma_pytorch.py                # PaliGemma + Gemma Expert 封装
├── preprocessing_pytorch.py        # 图像预处理（PyTorch 实现）
└── transformers_replace/           # HuggingFace transformers 补丁（adaRMSNorm 支持）
    └── models/
        ├── gemma/
        │   ├── configuration_gemma.py
        │   └── modeling_gemma.py
        ├── paligemma/
        │   └── modeling_paligemma.py
        └── siglip/
            ├── check.py
            └── modeling_siglip.py

scripts/
└── convert_jax_to_pytorch.py      # JAX checkpoint → PyTorch safetensors 转换脚本
```

### 使用步骤

#### 1. 安装依赖

确保已安装 `torch`、`transformers==4.53.2`、`safetensors`。

然后需要将 `transformers_replace/` 中的补丁文件复制到已安装的 transformers 包目录下：

```bash
# 找到 transformers 安装路径
python -c "import transformers; print(transformers.__file__)"

# 复制补丁文件（覆盖原文件）
cp training/models_pytorch/transformers_replace/models/gemma/configuration_gemma.py <transformers_path>/models/gemma/
cp training/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py <transformers_path>/models/gemma/
cp training/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py <transformers_path>/models/paligemma/
cp training/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py <transformers_path>/models/siglip/
cp training/models_pytorch/transformers_replace/models/siglip/check.py <transformers_path>/models/siglip/
```

> **为什么需要打补丁？** PI05 的 action expert 使用 adaRMSNorm 注入 flow-matching timestep，HuggingFace transformers 原生不支持此功能，需要替换相关文件。

#### 2. 转换 JAX 权重为 PyTorch 格式

```bash
cd /path/to/robotera_vla_drobotics

python scripts/convert_jax_to_pytorch.py \
    --checkpoint_dir /path/to/jax_checkpoint \
    --config_name pi05_M7_pp_opensource \
    --output_path /path/to/pytorch_output
```

转换后在 `output_path` 下生成 `model.safetensors`。

#### 3. 使用 PyTorch 推理

当 checkpoint 目录下存在 `model.safetensors` 文件时，`create_trained_policy()` 会自动检测并加载 PyTorch 模型：

```python
from training.interfaces.policies.policy_config import create_trained_policy
from training.configs.config import get_config

train_config = get_config("pi05_M7_pp_opensource")
policy = create_trained_policy(
    train_config,
    checkpoint_dir="/path/to/pytorch_output",  # 包含 model.safetensors
    pytorch_device="cuda:0",
)
result = policy.infer(obs)
```

### 支持的模型配置

- `pi05=True, no_state=True, action_dim=38, action_horizon=20`（M7 双臂）
- 完整支持 PI05 的 adaRMSNorm、time_mlp_in/out 架构
- 兼容 PI0 模式（state_proj + action_time_mlp）

## License

This repository is released under the MIT License. See `LICENSE`.
