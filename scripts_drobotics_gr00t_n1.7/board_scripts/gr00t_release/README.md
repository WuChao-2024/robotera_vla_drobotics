# gr00t_release — GR00T-M7 板端 BPU 推理服务(最终发布版)

GR00T-N1.7(cam_20m/checkpoint-150000)三塔 BPU int8 推理服务,action 塔为融合版(state/action encoder、norm、Euler 全部编译进图)。纯计算稳态 97ms(ema 101),单次分解约 102ms:act_bpu 51.8 + vision_bpu 17.7 + lang_bpu 17.5 + CPU 预处理杂项。

协议与 star1-benchmark 评测端(vla_test_xbot)兼容:ZMQ REP,收 NPZ(images 三摄 + state 57 维 + text),回 actions (N,16,38),base 系绝对末端位姿 xyzw。

## 启动

```bash
bash run_serve.sh [端口]   # 默认 5558
```

日志在同目录 serve.log,每次 query 打印 t=xx ms(ema)与分段耗时。

## 目录结构

代码与启动脚本在本目录,全部模型资产在 bpu_model/v1/(换版本时整个 v1 目录替换即可,代码不变)。

本目录:

| 文件 | 说明 |
|---|---|
| serve_m7_eval_bpu_fused_opt.py | 推理服务主程序。vision 三摄 n_task=3 异步分核 + IONArray 零拷贝绑定 + 文本预处理按指令缓存 + CPU 管线重排,数值与未优化版对拍逐位一致。URDF 兜底路径已指向 bpu_model/v1 |
| run_serve.sh | 启动脚本,用法 bash run_serve.sh 端口,模型路径已指向 bpu_model/v1 |
| camera_frame.py | cam_absolute 与 fk_base_rs 的 numpy 实现,把相机系 SE(3) delta 还原为 base 系绝对位姿 |
| board_preprocess.py | 文本到 input_embeds/position_ids/mask 的运行时预处理 |
| requirements.txt | 依赖版本清单 |

bpu_model/v1:

| 文件或目录 | 说明 |
|---|---|
| vision.hbm | 视觉塔 BPU 模型(int8) |
| language.hbm | 语言塔 BPU 模型(int8) |
| action_fused.hbm | 动作塔 BPU 模型(融合版,encoders/norm/Euler 进图,DiT 迭代去噪在图内) |
| embed_tokens.npy | 语言塔 embedding 查表,形状 151936x2048,fp16 |
| statistics.json | 归一化统计量(state q01/q99 + relative_action min/max + action q01/q99) |
| cosmos | Cosmos-Reason2-2B 的 tokenizer 与 chat_template,文本运行时预处理用 |
| l3_4.urdf | 腰颈正运动学用 URDF,run_serve.sh 与主程序兜底均指向本副本 |

## 运行环境

系统 python3.12(实测运行环境)。核心依赖:pyzmq、numpy、scipy、opencv-python、transformers、torch,版本见 requirements.txt。

## 数值与延迟口径

action_fused.hbm 与三塔分离版在 60 包回灌对拍下逐位一致(融合引入误差由上游转换链控制在 cos 0.99944);优化项(分核异步、零拷贝、缓存、重排)均为纯调度改动,不改数值。97ms 为纯计算稳态(端到端含 ZMQ/序列化另计)。
