# Docker project template

```bash
# Pull the pytorch image
docker pull adrianorenstein/pytorch:latest

# Run the image with:
make run pytorch

# Jupyter lab
make jupyter
```

If you want to use the pre-commit hook, run `pre-commit install` inside of the dev container. 


# SIF
Converting docker images into apptainer sifs
```bash

# on the cluster
apptainer pull docker://adrianorenstein/atari_pytorch:latest
# apptainer run ./converted_on_cluster_atari_pytorch_latest.sif python apptainer_images/benchmark_mmul.py
# Average time: 3.882581 seconds
# Standard deviation: 0.039912 seconds

# or locally, then copy to cluster
echo "building sif files"
make build apptainer && \
    docker pull --platform linux/amd64 adrianorenstein/pytorch:latest && \
    docker save -o pytorch_amd64.tar adrianorenstein/pytorch:latest && \
    make run apptainer apptainer build pytorch_amd64.sif docker-archive://pytorch_amd64.tar
```
