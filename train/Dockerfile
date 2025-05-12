FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC


RUN pip install ftfy regex

# Install system packages
RUN apt update && apt install -y \
    python3 python3-dev python3-pip python3-setuptools \
    git curl ffmpeg \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Symlink for convenience
RUN ln -s /usr/bin/python3 /usr/bin/python && python --version

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch (CUDA 12.1)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install OpenMMLab dependencies (for PlantSeg + MMSeg models)
RUN pip install -U openmim && \
    mim install 'mmengine>=0.10.0' && \
    mim install 'mmcv==2.1.0'

# Install JupyterLab, Ray, MLflow, and common utils
RUN pip install jupyterlab ray[default]==2.7.1 mlflow \
    matplotlib seaborn opencv-python tqdm albumentations

# Set default workdir
WORKDIR /workspace
