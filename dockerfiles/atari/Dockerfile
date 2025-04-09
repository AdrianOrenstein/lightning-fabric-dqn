ARG PLATFORM
ARG base_image=adrianorenstein/pytorch:latest
FROM --platform=${PLATFORM:-linux/amd64} ${base_image}

RUN pip install ale-py opencv-python-headless lz4

RUN git config --global --add safe.directory /app

ENV PYTHONPATH="/app/"

RUN useradd -u 1000 -m ubuntu
WORKDIR /app
USER ubuntu

CMD [ "/bin/bash" ]