FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:apptainer/ppa && \
    apt-get update && \
    apt-get install -y apptainer && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

CMD [ "/bin/bash" ]

