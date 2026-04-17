# ========== 基础镜像 ==========

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONPATH=/workspace
ENV PATH=/opt/conda/bin:$PATH

ARG XBOT_CUDA_VERSION=cu128
ARG XBOT_PYTORCH_VERSION=2.7.1

# ========== 安装系统依赖 ==========
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list \
    && sed -i 's|http://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3.10-distutils python3.10-venv \
        netcat dnsutils gnupg2 lsb-release software-properties-common \
        zip unzip wget curl git git-lfs build-essential cmake \
        vim less sudo htop ca-certificates man tmux ffmpeg \
    && rm -rf /var/lib/apt/lists/*


# ========== 创建 Python 3.10 环境 ==========
RUN conda create -y -n py310 python=3.10 pip setuptools \
    && conda clean -a
ENV CONDA_DEFAULT_ENV=py310
ENV PATH=/opt/conda/envs/py310/bin:$PATH

# 升级 pip/setuptools/wheel
RUN python -m pip install --upgrade pip setuptools wheel

# ========== 安装 PyTorch ==========
RUN python -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128

# ========== 添加 ROS 2 Humble apt 源 ==========
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    > /etc/apt/sources.list.d/ros2.list

# ========== 安装 ROS2 Humble ==========
RUN apt-get update && apt-get install -y \
        ros-humble-desktop \
        ros-humble-rmw-cyclonedds-cpp \
        python3-colcon-common-extensions \
        python3-rosdep \
        python3-lark \
    && rm -rf /var/lib/apt/lists/*

# 初始化 rosdep
RUN rosdep init || true \
 && rosdep update || true

# 设置 ROS 2 环境变量
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

ENV PATH=/opt/ros/humble/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/ros/humble/lib:$LD_LIBRARY_PATH
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# ========== 设置工作目录 ==========
WORKDIR /workspace

# COPY ./pyproject.toml .
COPY . .

    
RUN pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple "." \
    --extra-index-url https://download.pytorch.org/whl/2.7.1

RUN cd $(python3 -c "import site; print(site.getsitepackages()[0])") && \
    ln -s transformations tf_transformations


RUN mv /opt/conda/envs/py310/lib/libstdc++.so.6 /opt/conda/envs/py310/lib/libstdc++.so.6.bak
RUN pip install --no-cache-dir \
    catkin_pkg \
    lark-parser \
    empy==3.3.4 \
    setuptools
RUN source /opt/ros/humble/setup.bash && colcon build --packages-select xbot_common_interfaces
RUN echo "source /workspace/install/setup.bash" >> ~/.bashrc

ENV ROS_DOMAIN_ID=211
ENV LD_LIBRARY_PATH=/opt/conda/envs/py310/lib:$LD_LIBRARY_PATH

# 默认进入 bash
CMD ["/bin/bash"]

CMD ["/bin/bash"]
