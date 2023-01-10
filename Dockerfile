FROM nvidia/cuda:11.4.0-cudnn8-runtime-ubuntu20.04

RUN apt update && apt upgrade -y && \
    apt install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt install python3.9 -y && apt install python3-pip -y && \
    apt install python3.9-dev -y

RUN apt-get update && apt-get install -y git 
RUN apt-get install -y libsm6 libxrender1 libfontconfig1 libxext6 libgl1-mesa-glx && \
    pip3 install -U pip setuptools wheel


RUN apt-get pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9/index.html

RUN git clone https://github.com/open-mmlab/mmcv.git /mmcv
WORKDIR /mmcv
RUN git rev-parse --short HEAD
RUN pip install --no-cache-dir -e .[all] -v && pip install pre-commit && pre-commit install

ENTRYPOINT [ "bash" ]