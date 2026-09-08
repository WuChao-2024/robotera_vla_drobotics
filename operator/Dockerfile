# robotera_vla M7 训练算子镜像 (norm_stats + train 两入口)
# 构建: operator/build_docker.sh 组装 staging context 后 buildx 构建
# 层序(为削空间峰值专门设计): apt -> 源码 -> venv(torch 层与项目依赖层分离)
#        -> 权重(12G, 最后) -> operator 三件套
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
ARG http_proxy
ARG https_proxy
ARG no_proxy=localhost,127.0.0.1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev python3-pip \
        build-essential git ca-certificates wget \
        libgl1 libegl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        libxkbcommon0 libjpeg8 zlib1g ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- 项目源码 (运行时通过薄拷贝树 import, 项目零修改) ----
COPY --chmod=a+rX src/ /app/src/

# ---- venv 层 1: torch cu128 (dataloader 栈) ----
RUN python3.12 -m venv /app/venv \
    && /app/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel \
    && /app/venv/bin/pip install --no-cache-dir \
         torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
         --index-url https://download.pytorch.org/whl/cu128 \
    && chmod -R a+rX /app/venv

# ---- venv 层 2: 项目全部依赖 (含 jax[cuda12]==0.5.3, lerobot, pinocchio) ----
RUN /app/venv/bin/pip install --no-cache-dir /app/src \
         --extra-index-url https://download.pytorch.org/whl/cu128 \
    && /app/venv/bin/python -c "import pinocchio" \
    && chmod -R a+rX /app/venv

# ---- 预置 gs:// 资产的本地缓存 (OPENPI_DATA_HOME 命中, 离线运行) ----
# pi05_base 基座权重 12G + paligemma tokenizer。
# 缓存树目录必须 0777: download.py 的 _set_permission 见权限已满足会跳过 chmod,
# uid 10001 非属主也不会 EPERM; COPY 自动建的中间目录是 root:0755, 故先 RUN 建
# 好 777 目录再 COPY(目录元数据 copy-up 仅 3 个条目, 不会复制 12G 内容)。
# 源文件 mtime(2026-08) 晚于缓存失效线, COPY 保留 mtime 即命中。
RUN mkdir -p /app/openpi_data/openpi-assets/checkpoints /app/openpi_data/big_vision \
    && chmod 0777 /app/openpi_data /app/openpi_data/openpi-assets /app/openpi_data/openpi-assets/checkpoints /app/openpi_data/big_vision
COPY --chmod=a+rwX weights/pi05_base/ /app/openpi_data/openpi-assets/checkpoints/pi05_base/
COPY --chmod=a+rwX weights/big_vision/ /app/openpi_data/big_vision/

# ---- operator 入口与适配器 ----
COPY --chmod=755 entry/ /app/entry/
COPY --chmod=a+rX adapters/ /app/adapters/

ENV PATH=/app/venv/bin:$PATH \
    OPENPI_DATA_HOME=/app/openpi_data \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp/home
RUN mkdir -p /tmp/home && chown -R 10001:10001 /tmp/home
