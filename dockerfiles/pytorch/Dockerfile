FROM python:3.13

ARG DEBIAN_FRONTEND=noninteractive
ENV DOCKER_RUNNING=true
ENV APPTAINER_RUNNING=true

RUN apt-get update && \
    apt-get install rpm2cpio cpio -y && \
    curl -s https://raw.githubusercontent.com/apptainer/apptainer/main/tools/install-unprivileged.sh | \
    bash -s - install-dir && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install \
    torch torchvision torchrl tensordict einops \
    numpy opt_einsum numba \
    pandas matplotlib seaborn scikit-learn jupyterlab ipywidgets \
    tyro tqdm loguru \
    ruff pre-commit \
    flake8-black flake8 isort \
    pytest pytest-benchmark pytest-parallel py coverage \
    redis wandb moviepy imageio pysocks 'requests[socks]' \
    lightning wandb lightning-bolts

RUN git config --global --add safe.directory /app

ENV PYTHONPATH="/app/"

RUN useradd -u 1000 -m ubuntu
WORKDIR /app
USER ubuntu

CMD [ "/bin/bash" ]