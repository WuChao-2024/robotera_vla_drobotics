# robotera_vla M7 训练算子工程

本目录将 robotera_vla 的 M7 pi0.5 训练能力封装为 droboticflow 平台算子，**不改动项目任何既有代码**。一个镜像承载两个功能入口，平台上注册为两个算子。端口契约 v1.1.0（assets 优先设计）：

```
工作流输入 ds-1..N (BOS 目录引用, 可在 UI 改选)
    │
    ├─► 目录读取-1 ──► M7 pi0.5 归一化统计 ── assets(第一输入) ─► M7 pi0.5 训练 ──► model
    ├─► 目录读取-2 ──►  (任意个 dataset 端口)                      (assets 第一必须,
    └─► 目录读取-N                                                 其余已连线端口皆数据集)
```

**端口契约 v1.1.0**：训练算子第 1 输入 `assets`（必须，连统计算子输出），第 2 输入 `dataset`（必须，一个 LeRobot 数据集根目录）；第 3 个起用户在工作流编辑器**直接新增输入端口并连线即可，端口无命名与数量限制，均视为数据集**（适配器只认 `assets` 一个锚点端口，其余已连线端口全部按数据集严格校验：必须有 meta/ 与 data/，否则 run 失败并给出明确指引；assets 连了但没有 norm_stats.json 同样直接失败）。统计算子只有一个必须的 `dataset` 端口，同样支持任意加端口。无任何超参回退，数据集唯一入口是端口连线。

## 平台产出索引（2026-09-03 验收完成）

| 产出 | 标识 | 内容 |
|---|---|---|
| 镜像 | robotera-vla-train:20260908_120126 | v1.1.0 契约镜像, 约 21GB, 两入口共用 |
| 算子 | M7 pi0.5 归一化统计 operator-c469763d61b63eac2fedd06fc5dffa89 v1.1.0 | dataset(必须)输入, assets 输出, 纯 CPU, 支持任意加端口 |
| 算子 | M7 pi0.5 训练 operator-e10fce51d2e3930427e592c65b04c3ce v1.1.0 | assets(第一必须)+dataset(第二必须)输入, model 输出, 支持任意加端口 |
| 工作流 | M7 pi0.5 端口连线回归 v2 workflow-41c70e4362dae82bf585e9ef3620cca3 | v1.1.0 模板: assets 第一, dataset + 2 个 UUID 端口连 3 个数据集, 内置冒烟参数 |
| 回归 run | run-2d3e2c43617bba976b109bf80dc39a72 completed | model 12441675812 字节, 35 分钟, UUID 端口实证生效 |
| 旧版工作流/run | workflow-83a724ea 等 v1.0.0 系列 | 2026-09-03 验收版, 端口为 dataset x4+assets 形态, 已被 v1.1.0 取代 |

验收 run 的训练节点资源为 cpu 32 / 内存 512Gi / GPU 1（工作流节点级覆盖）；算子注册体默认 memoryLimit 96Gi 偏小（orbax 异步保存会被 OOMKill），新建工作流时务必在节点级给 512Gi，`platform_bootstrap.py workflow` 默认即如此。

现网镜像 20260903_200641 内仍残留已废除的 DATASET_PATH 回退分支：属死代码，仅在"无 dataset 端口连线"时触发并报错退出，端口连线路径完全不经过；下次重建镜像自然清除。平台算子定义已删除该超参（2026-09-07 更新），数据集唯一入口为 dataset 端口连线。

## 目录结构

```
operator/
├── Dockerfile                  # 镜像定义（层序专为构建机磁盘与 PVC 特性设计）
├── build_docker.sh             # 组装 staging context 并后台构建推送
├── entry/
│   ├── norm_stats.sh           # 归一化统计入口
│   └── train.sh                # 训练入口（全部参数可超参覆盖）
├── adapters/
│   ├── gen_data_config.py      # 由 dataset 端口清单或目录扫描，运行时生成 data_config yml
│   ├── prepare_inputs.py       # 解析上游连线与 artifacts: dataset 端口集合 + assets 里的 norm_stats
│   └── finalize.py             # 收尾协议：result.json + stage-manifest + 产物上报
├── operator-definitions/
│   ├── norm_stats_operator.json
│   └── train_operator.json     # 算子注册体（端口/超参/资源）
├── platform_tools/
│   └── platform_bootstrap.py   # 注册算子、建工作流、发起 run 的命令行工具
└── selfcheck/
    └── run_selfcheck.sh        # 出厂自检（uid 10001 运行镜像）
```

## 构建镜像

前置条件：

- 构建机装有 docker 与 buildx（docker-container 类型 builder）
- 本机已有官方 pi05_base 权重缓存（约 12GB，目录内有 `pi05_base/params` 与 `big_vision/paligemma_tokenizer.model`），默认路径见 build_docker.sh 的 WEIGHTS_SRC，可用环境变量覆盖
- 网络代理可用（pip / apt / git 均走代理）

```bash
cd operator
bash build_docker.sh
```

脚本行为：

1. 在 /tmp 组装 staging（项目源码排除 docs、scripts_drobotics* 等 + operator 三件套 + 权重），项目文件零改动
2. 后台调用 buildx 构建并推送，镜像约 28GB（含 12GB pi05_base），tag 自动取时间戳
3. 日志在 /tmp/rv_build.log，tag 写 /tmp/rv_cur_tag，构建约 30~45 分钟（层缓存命中时显著缩短）

Dockerfile 层序的设计约束（改动前必读）：

| 约束 | 原因 |
|---|---|
| 权重层必须放在 venv 依赖层之后 | pip 峰值约 15GB，与 12GB 权重层叠加会写爆构建盘 |
| 严禁对 12G 权重层做 RUN touch/chmod/chown | overlay copy-up 会把整层复制一份导致爆盘 |
| 权重目录权限用 COPY --chmod=a+rwX，且中间目录先 RUN 建好 0777 | openpi 的 download.py 对缓存根目录做 chmod，权限已满足时跳过；否则 uid 10001 报 EPERM |
| 基座 mtime 晚于 2025-02-03 即可命中缓存 | COPY 保留源 mtime，download.py 的失效线在该日期之前 |

## 注册算子与建工作流

platform_tools/platform_bootstrap.py 封装了平台 API 调用（地址与账号在脚本内配置或自行修改，本文档不含凭证）：

```bash
# 注册两个算子；--image 为刚构建的完整 tag。平台上已存在同名算子时默认跳过，
# 加 --update 则用本地 JSON 刷新平台上的算子定义（含镜像 tag）
python3 platform_tools/platform_bootstrap.py register --image <registry>/robotera-vla-train:<tag> --update

# 创建项目 + 验收形态工作流：每个子数据集一个「目录读取」节点
# --dataset-paths 逗号分隔 BOS 路径（bos://<bucket>/<prefix>），生成目录读取节点、
# dataset 端口连线、工作流输入三件套；不传则建无目录读取的两节点工作流
WORKFLOW_TRAIN_GPU=1 WORKFLOW_TRAIN_MEM=512Gi \
python3 platform_tools/platform_bootstrap.py workflow \
  --dataset-paths bos://testxyz/cauchy_dev/robotera_opensource/M7_pickplace_example/2031605,bos://testxyz/cauchy_dev/robotera_opensource/M7_pickplace_example/2031607,bos://testxyz/cauchy_dev/robotera_opensource/M7_pickplace_example/2031704 \
  --node-timeout 10800 --run-timeout 14400

# 发起 run；--node-overrides 传节点级超参覆盖
python3 platform_tools/platform_bootstrap.py run \
  --node-overrides '{"train-node": {"hyperparams": [{"key": "NUM_TRAIN_STEPS", "value": "20"}]}}'

# 查看已注册的 M7 算子与项目下工作流
python3 platform_tools/platform_bootstrap.py status
```

注意：

- 平台工作流节点固化创建时的镜像 tag，镜像更新后需要 register --update 刷新算子并重建工作流（或 run 时用 image 覆盖）
- 平台 run 发起接口的 nodeOverrides 字段会被静默忽略（前端只发 trigger 与 inputs），节点级参数覆盖的官方做法是把超参直接写进工作流节点定义（PUT /api/workflows/{id} 修改 node.data.hyperparams 后再发起 run）
- 平台内置「目录扫描」算子 2026-09 起注册表定义为仅 json 概览输出（无 directory 输出端口），builder 自动改用快照复刻的 legacy 目录读取节点（平台按节点冻结算子快照执行，与旧验收工作流同机制）
- 工作流输入只能绑定无入边的根节点；训练节点因有 assets 入边不能直接接工作流输入，数据集必须经目录读取节点中转
- 目录读取节点需要存储连接（脚本内 STORAGE_CONNECTION_ID / STORAGE_BUCKET 常量），其必填 sourceDirectory 由工作流输入的 BOS 目录引用默认值喂入，UI 发起 run 时可改选其他目录
- 有活跃 run 时平台拒绝建新 run（409），先 stop 再发

## 已验证的训练参数与性能（A800-80G 节点）

验收 run（run-3239817171eb921471ab06eb52d662b5，completed）的冒烟配置与实测：

| 参数 | 值 | 说明 |
|---|---|---|
| GPU | 1 x A800-80G | 单卡可跑通全链路 |
| FSDP_DEVICES / BATCH_SIZE | 1 / 2 | 单卡全参微调最小可行配置 |
| NUM_TRAIN_STEPS / SAVE_INTERVAL | 20 / 10 | 冒烟步数 |
| 产物 | model 12.4GB | orbax checkpoint, 与官方 M7_pickplace_example_ckpt 同构 |
| 实测速度 | 10.7 s/it | 单卡 batch=2 |
| 显存 | XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 | 0.8 会在单卡全参时 OOM（需要 64.4GB） |
| 容器内存 | 节点级 512Gi | orbax 异步保存时 host 端缓冲 50GB+，96Gi 会被 OOMKill |
| checkpoint 目录 | /tmp/ckpt（容器本地盘） | 单份含 train_state 约 50GB，写 200G 的 workspace PVC 会挤爆；交付物由 finalize 拷回 OUTPUT_DIR |
| 权重加载 | 镜像内置 pi05_base 缓存命中 | NoActionWeightLoader 自动重初始化 action_in/out_proj（38 维） |

官方基准（README 声明，未在本平台复跑）：8 x A800，fsdp=8，batch=64，约 1.8 s/it；300k 步按此速度约 150 小时，官方文档的 24 小时对应更少步数量级（config 注释含 300k/4=75k 的缩法）。正式训练按需取 NUM_TRAIN_STEPS 与 WORKFLOW_TRAIN_GPU（默认 8），并把 runTimeoutSeconds 按 步数 x 单步耗时 上调并留缓冲。

归一化统计算子：纯 CPU 资源即可；调试可用 SAMPLE_RATIO=0.001~0.01 与 MAX_FRAMES 控制时长，全量默认 SAMPLE_RATIO=0.1（验收 run 即全量统计，3 个子数据集约 12 分钟）。

## 算子运行协议（对接平台的关键行为）

- 入口只做三件事：准备运行时工作树（源码实体拷贝到 OUTPUT_DIR/work，动态生成 data_config 覆盖副本）、调用训练程序、调用 finalize 收尾
- 输入解析走 prepare_inputs.py：优先读 INPUT_BINDINGS_PATH（平台按本节点输入端口分组的连线清单，targetPortId 为 dataset/dataset_N/assets），无 bindings 时回退扫描 INPUT_ARTIFACTS_PATH 的 portId；上游目录路径 bos://testxyz/&lt;key&gt; 映射到 /workspace/&lt;key&gt; 本地读（workspace PVC 即该桶的 CSI 挂载）
- dataset 端口路径必须本地可达且为合法 LeRobot collection（含 meta/ 与 data/），否则入口显式报错退出
- assets 端口收到 norm_stats.json 时训练直接复用（ASSETS_READY=1）；缺失且 AUTO_NORM_STATS=1 时按 SAMPLE_RATIO 自动补算
- finalize 生成 result.json（artifacts 的 id 采用 portId-NodeId 形式防全局重名）、stage-manifest.json，并打印 DROBOTICFLOW_RESULT_PATH 行
- 同 run 内 pod 重试（RUN_ATTEMPT 大于 1）自动切 --resume；但默认 checkpoint 在容器本地盘，跨 pod 不保留，需要续训语义时把 CKPT_DIR 指向 PVC 路径
- 失败时以非 0 退出，不做收尾

## 踩坑记录（平台环境特性）

| 现象 | 根因与对策 |
|---|---|
| cp -rs 在输出目录批量报 File exists | workspace PVC（csi-bos）对批量 symlink 创建不友好，源码仅 40MB 改实体拷贝 |
| PermissionError Operation not permitted 于缓存目录 | openpi 下载缓存目录链需要 0777（见构建节），非属主 chmod 被跳过的前提 |
| 训练在异步保存 checkpoint 时静默死亡 | 容器内存 limit 不足被 OOMKill，节点级上调至 512Gi |
| 单卡 RESOURCE_EXHAUSTED | XLA 显存比例 0.8 上限 64G，单卡全参需 64.4G；上调 0.95 |
| 上游产物 portId 对不上本节点输入端口 | INPUT_ARTIFACTS 里的 portId 是上游输出端口名，必须按 INPUT_BINDINGS_PATH 的 targetPortId 归组解析 |
| 监控页 artifacts 列表极长 | 目录读取算子把目录内每个文件都列为文件级 artifacts（千级），平台固有行为，不影响功能 |
| 调试 run 长时间无产物 | 排查手段：平台 run 日志接口 + kubectl 实时抓 pod 日志（失败 pod 会被快速回收） |
| run 显示 failed 但训练其实完成 | 平台产物外部化（12GB 模型上传 BOS）在集群高负载时段可能以 BOS PreconditionFailed(If-Match) 中断，K8s Job 实际 Completed、BOS 上留部分文件；节点记录 startedAt/endedAt 失真、run 级状态长时间显示 queued 也不可信。对策：重发 run（低负载时段成功率高），判定真伪看 kubectl events 与 BOS 目录 |
