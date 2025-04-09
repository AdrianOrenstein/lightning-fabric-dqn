FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG PYTHON_VERSION=3.13
ARG DEBIAN_FRONTEND=noninteractive
ENV DOCKER_RUNNING=true
ENV APPTAINER_RUNNING=true
ENV TZ=US

RUN apt-get -y update -qq && apt-get install software-properties-common -y && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y \
    python${PYTHON_VERSION}-full \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python-setuptools \
    python-pip \
    build-essential \
    ca-certificates \
    clang \
    gcc \
    cmake \
    htop \
    curl \
    git \
    libomp-dev \
    libsm6 \
    libssl-dev \
    libxrender-dev \
    libxext-dev \
    iproute2 \
    vim \
    ssh \
    wget \
    vim \
    zip \
    && \
    wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq && \
    chmod +x /usr/bin/yq && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION}

RUN pip install \
    torch torchvision torchrl tensordict \
    ale-py opencv-python-headless lz4 \
    stable-baselines3 \
    tyro tqdm loguru \
    numpy opt_einsum numba \
    pandas matplotlib seaborn scikit-learn jupyterlab ipywidgets \
    ruff pre-commit \
    pytest pytest-instafail pytest-benchmark "pytest-xdist[psutil]" py pytest-cov \
    redis pygame protobuf moviepy imageio pysocks 'requests[socks]' \
    lightning lightning-bolts wandb tensorboard

RUN git config --global --add safe.directory /app

ENV PYTHONPATH="/app/"

RUN useradd -u 1000 -m ubuntu
WORKDIR /app
USER ubuntu

CMD [ "/bin/bash" ]