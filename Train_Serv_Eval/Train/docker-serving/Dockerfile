# ✅ 基础镜像：包含 Python 3.10 + CUDA 11.8 + PyTorch 2.1.0
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# ✅ 避免交互式命令提示
ENV DEBIAN_FRONTEND=noninteractive

# ✅ 设置工作目录
WORKDIR /app

# ✅ 安装系统依赖（OpenCV + PNG 支持）
RUN apt-get update && apt-get install -y \
    git wget curl libgl1 unzip \
    libgl1-mesa-glx \
    libglib2.0-0  libsm6 libxext6 libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ✅ 安装 Python 依赖（FastAPI + 推理 + 图像）
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    onnxruntime-gpu \
    prometheus-fastapi-instrumentator \
    numpy \
    opencv-python-headless \
    matplotlib \
    pillow \
    regex \
    ftfy \
    yipy \
    openmim \
    onnx \
    torchinfo   
#    && rm -rf /root/.cache/pip


# ✅ Install mmengine + mmcv
#RUN mim install mmengine \
#   && mim install "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html \



COPY mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl /tmp/

RUN pip install mmengine && \
    pip install /tmp/mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl && \
    rm -rf /root/.cache/pip


RUN git clone -b main https://github.com/open-mmlab/mmsegmentation.git /app/mmsegmentation && \
    pip install -v -e /app/mmsegmentation

#   && rm -rf /root/.cache/pip
# ✅ Clone and install MMSegmentation + PlantSeg
#RUN git clone -b main https://github.com/open-mmlab/mmsegmentation.git \
#   && cd mmsegmentation && pip install -v -e . && cd .. \
#   && git clone https://github.com/tqwei05/PlantSeg.git
# ✅ 拷贝服务脚本（或由 docker-compose volumes 动态挂载）
COPY app.py .

# ✅ 显式声明服务端口
EXPOSE 8000

# ✅ 启动 FastAPI 服务（适用于生产环境）
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
