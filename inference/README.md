# Robotera VLA Inference Module

[![OS](https://img.shields.io/badge/OS-Ubuntu%2022.04-orange.svg)](https://ubuntu.com/)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](https://www.docker.com/)
[![GPU](https://img.shields.io/badge/NVIDIA-CUDA%2012.4-green.svg)](https://developer.nvidia.com/cuda-zone)

This module defines the deployment protocol for the **Robotera Vision-Language-Action (VLA)** model on an Inference PC and its integration with robot-side applications via **ROS 2 Humble**.

## Inference Machine & Environment

The inference module is designed to run on a dedicated **Inference PC** equipped with an NVIDIA GPU. The system architecture consists of two main components:

| Component | Description |
|-----------|-------------|
| **Robot Side** | Runs ROS 2 Humble, captures observations (camera images, robot state), and executes generated actions |
| **Inference PC** | Hosts the VLA model inside a Docker container, processes observations, and outputs actions |

### Hardware Requirements

| Resource | Minimum Specification |
|----------|----------------------|
| **GPU** | NVIDIA GPU with at least 12GB VRAM (e.g., RTX 4070/5070) |
| **Storage** | 20GB+ free space |
| **Network** | Gigabit Ethernet (low-latency connection to robot) |

### Software Stack
```text
┌─────────────────────────────────────────────────────────┐
│ Inference PC                                            │
│ ┌───────────────────────────────────────────────────┐   │
│ │ Docker Container                                  │   │
│ │ ┌─────────────┐  ┌─────────────────────────────┐  │   │
│ │ │ VLA Model   │  │ Python Inference Script     │  │   │
│ │ │ (JAX)       │◄─┤ • ROS 2 Subscriber (obs)    │  │   │
│ │ └─────────────┘  │ • ROS 2 Publisher (action)  │  │   │
│ │                  └─────────────────────────────┘  │   │
│ └───────────────────────────────────────────────────┘   │
│ ↕ ROS 2 (DDS over UDP)                                  │
└─────────────────────────────────────────────────────────┘
                                ↕
┌─────────────────────────────────────────────────────────┐
│ Robot                                                   │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐         │
│ │ Cameras     │ │ Robot State │ │ Actuators   │         │
│ └─────────────┘ └─────────────┘ └─────────────┘         │
│ ROS 2 Humble                                            │
└─────────────────────────────────────────────────────────┘
```

## Scope
---
- **Deployment**: Standardized environment setup using Docker for high-performance inference.
- **Protocol**: ROS 2 topic/service contracts for seamless Robot-to-PC communication.
- **Development**: Client and server stubs for rapid prototyping and testing.

---

## Environment Setup (Docker)

We provide a specialized `inference.Dockerfile` that serves as a **complete, turnkey environment**.

### 1. Prerequisites
Ensure your host machine has the following installed:
* **Docker Engine**: [Official Installation Guide](https://docs.docker.com/engine/install/ubuntu/)
* **NVIDIA Container Toolkit**: [Official NVIDIA-SMI Runtime Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### 2. Build the Inference Image
Execute the following command from the project root:
```bash
docker build -t m7_vla:v1 -f inference.Dockerfile .
```
Execute the following command to build your docker
```bash
docker run -d \
    --name robotera-vla \
    --network=host \
    --gpus all \
    -it \
    --restart=always \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd)/checkpoints:/workspace/checkpoints \
    m7_vla:v1
```

## Inference Steps

We provide a pretrained checkpoint for a simple pick-and-place scenario. You can download it from [https://huggingface.co/roboterax/M7_pickplace_example_ckpt]. Below is an example to demonstrate how to run inference.




### 1. Command in robot shell

```bash

ssh developer@192.168.8.100 -p 2222      

# source 消息定义
git clone https://github.com/roboterax/teleop_client.git
cd teleop_client
colcon build
source install/setup.bash

# 启动关节服务
ros2  service call /dynamic_launch xbot_common_interfaces/srv/DynamicLaunch "app_name: ''
sync_control: false
launch_mode: 'pos'"

# 初始化关节模组，此过程需要大概20秒M7整机关节使用
ros2  service call /ready_service std_srvs/srv/Trigger {}

# 小臂抬起
ros2 action send_goal /simple_actions xbot_common_interfaces/action/SimpleActions "{action_name: 'lift_up', time_cost: 4.0}"

ros2 service call /activate_service std_srvs/srv/Trigger {}

ros2 service call /teleoperation/service xbot_common_interfaces/srv/StringMessage '{data: "{\"type\": \"mpc\", \"message\": \"{\\\"command\\\": \\\"start\\\"}\"}"}'

ros2 service call /teleoperation/service xbot_common_interfaces/srv/StringMessage '{data: "{\"type\": \"webxr\", \"message\": \"{\\\"command\\\": \\\"start\\\", \\\"camera_type\\\": \\\"realsense\\\"}\"}"}'

ros2 service call /teleoperation/service xbot_common_interfaces/srv/StringMessage '{data: "{\"command\": \"startPushVideo\", \"message\": \"{\\\"video\\\": {\\\"head\\\": {\\\"width\\\": 848, \\\"height\\\": 480, \\\"fps\\\": 30, \\\"bitrate\\\": 4000000}, \\\"left\\\": {\\\"width\\\": 640, \\\"height\\\": 480, \\\"fps\\\": 30, \\\"bitrate\\\": 4000000}, \\\"right\\\": {\\\"width\\\": 640, \\\"height\\\": 480, \\\"fps\\\": 30, \\\"bitrate\\\": 4000000}}}\"}"}'


```

### 2. Command in PC shell 

Allow all users to connect to the X server (for GUI forwarding)
```bash
xhost +
```
Access the bash shell of the running Docker container "robotera-vla" in interactive mode
```bash
docker exec -it robotera-vla /bin/bash
```
Run the main inference script for the VLA model example
```bash
python inference/example/main.py
```
You can modify the inference settings such as checkpoint path, language instructions, and other parameters in the inference/example/config.py

The final result is roughly as shown below:

![Inference result](docs/PAP.gif)

*Figure 1: Inference result of the pick-and-place task.*