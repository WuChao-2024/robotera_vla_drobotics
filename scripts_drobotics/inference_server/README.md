# Pi05 推理服务 & JAX/PyTorch 对比工具

## 架构

```
compare.py (ZMQ 客户端)
  ├── ZMQ REQ → server.py :8003 (JAX)  → actions_jax
  └── ZMQ REQ → server.py :8004 (Torch) → actions_torch
       │
       └── 对比 → CSV / JSON
```

## 前置条件

### 环境

```bash
conda activate robotera_vla_drobotics
```

依赖：`pyzmq`、`numpy`、`pyarrow`、`opencv-python`（均在 `robotera_vla_drobotics` 环境中已有）。

### PyTorch 权重转换

Torch 推理服务需要 `model.safetensors` 格式的权重，需先运行转换脚本：

```bash
cd /home/chao.wu/pi_oellm/ROBOTREA/robotera_vla_drobotics
PYTHONPATH=$(pwd) python scripts/convert_jax_to_pytorch.py \
    --checkpoint_dir /home/chao.wu/pi_oellm/ROBOTREA/M7_pickplace_example_ckpt \
    --config_name pi05_M7_pp_opensource \
    --output_path /home/chao.wu/pi_oellm/ROBOTREA/M7_pickplace_example_ckpt_pt \
    --precision bfloat16
```

转换后在 `M7_pickplace_example_ckpt_pt/` 下生成 `model.safetensors` 和 `assets/`。

### transformers 补丁

PyTorch 推理需要将 adaRMSNorm 补丁复制到 transformers 安装目录，参见项目根 `README.md` 的「PyTorch 推理支持」章节。

## 使用方法

### 1. 启动 JAX 推理服务

```bash
cd /home/chao.wu/pi_oellm/ROBOTREA/robotera_vla_drobotics/scripts_drobotics/inference_server

python server.py --backend jax --port 8003 --warmup
```

默认参数即可使用，无需额外配置：
- checkpoint: `M7_pickplace_example_ckpt`
- policy: `pi05_M7_pp_opensource`
- prompt: `"pick the apple and put it in the bowl."`

### 2. 启动 PyTorch 推理服务

```bash
python server.py --backend torch --port 8004 --warmup
```

### 3. 运行对比

```bash
python compare.py
```

默认对比前 100 帧，输出到 `./comparison_results/`。

```bash
# 自定义帧数
python compare.py --frame-count 500

# 指定输出目录
python compare.py --output-dir ./results_20260603
```

## 输出文件

| 文件 | 说明 |
|------|------|
| `per_sample.csv` | 每帧的详细指标 |
| `aggregate.json` | 聚合统计（mean/std/min/max/median） |
| `worst_samples.csv` | MSE 最大的 20 帧 |

## 对比指标

| 指标 | 说明 |
|------|------|
| MSE | 均方误差 |
| RMSE | 均方根误差 |
| MAE | 平均绝对误差 |
| Max Abs Error | 最大绝对误差 |
| Relative Error | 相对误差 |
| Cosine Similarity | 余弦相似度（每时间步取均值） |
| Agreement @ 1e-N | 逐元素误差 < 10^-N 的比例 |

## ZMQ 协议

兼容 `star1_vla_inference` 协议，并扩展了 `noise` 字段用于确定性对比。

**请求**（NPZ 压缩字节）：
| 字段 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `images` | dict of uint8 | HWC | cam_high, cam_left_wrist, cam_right_wrist |
| `state` | float32 | (57,) 或 (B, 57) | 机器人状态 |
| `text` | str | — | 语言指令 |
| `noise` | float32（可选） | (20, 38) | 确定性噪声 |

**响应**（send_pyobj）：
| 字段 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `actions` | float32 | (B, 20, 38) | 动作 chunk |
