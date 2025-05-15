#  Base image: PyTorch 2.1.0 + CUDA 11.8 + Python 3.10
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

USER root

#  Install OS dependencies
RUN apt-get update && apt-get install -y \
    git wget curl libgl1 unzip \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

#  Upgrade pip
RUN pip install --upgrade pip

#  Python dependencies
RUN pip install --no-cache-dir \
    jupyterlab \
    torchinfo \
    regex \
    ftfy \
    yipy \
    pillow \
    openmim \
    onnx \
    onnxruntime-gpu \
    neural-compressor \
    numpy \
    tensorrt==8.6.1

#  Copy and install mmcv from local .whl file
COPY mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl .
RUN mim install mmengine \
 && pip install mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl

#  Clone and install MMSegmentation
RUN git clone -b main https://github.com/open-mmlab/mmsegmentation.git \
 && cd mmsegmentation && pip install -v -e . && cd ..

#  Clone and install PlantSeg
RUN git clone https://github.com/tqwei05/PlantSeg.git \
 && cd PlantSeg && pip install -v -e . && cd ..

#  Set working directory
WORKDIR /workspace
ENV JUPYTER_ENABLE_LAB=yes
RUN echo 'export PS1="\u@\h:\w\$ "' >> /etc/bash.bashrc

#  Expose Jupyter port
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token=''"]
