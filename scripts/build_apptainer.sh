#!/bin/bash

# Specify the image type (pytorch, minigrid, or atari) as a script argument
IMAGE_TYPE=$1

# Parse the values from the combined config.yaml file
USERNAME=$(yq e '.common.USERNAME' config.yaml)
IMAGE_NAME=$(yq e ".images.apptainer.IMAGE_NAME" config.yaml)
TAG=$(yq e ".images.apptainer.TAG" config.yaml)

FULL_IMAGE_NAME="$USERNAME/$IMAGE_NAME:$TAG"

# Debug: Check Dockerfile path
DOCKERFILE_PATH="dockerfiles/$IMAGE_TYPE/Dockerfile"
echo "Building with Dockerfile: $DOCKERFILE_PATH"

PLATFORMS=$(yq e ".images.$IMAGE_TYPE.PLATFORMS" config.yaml)
echo "Building for platforms: $PLATFORMS"

# make sure buildx is installed, and alias "docker build" to "docker buildx"
docker buildx install 

# build image for multiple platforms, push latest to docker.io
docker buildx build -t $FULL_IMAGE_NAME \
    --platform="$PLATFORMS" \
    -f dockerfiles/${IMAGE_TYPE}/Dockerfile . \
    --push

# and then load image into local registry
# you should see the image with "docker image ls" 
echo loading $FULL_IMAGE_NAME into docker image registry
docker build -t $FULL_IMAGE_NAME \
    -f dockerfiles/$IMAGE_TYPE/Dockerfile . \
    --load