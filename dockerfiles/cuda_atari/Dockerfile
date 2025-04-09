ARG PLATFORM
ARG base_image=adrianorenstein/cuda_pytorch:latest
FROM --platform=${PLATFORM:-linux/amd64} ${base_image}

RUN pip install ale-py opencv-python-headless lz4

CMD [ "/bin/bash" ]