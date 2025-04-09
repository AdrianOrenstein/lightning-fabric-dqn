#!/bin/bash

# Install yq if it's not already installed
if ! command -v yq &> /dev/null
then
    echo "yq could not be found, goto https://github.com/mikefarah/yq#install"

    if [[ "$OSTYPE" == "darwin"* ]]; then
        # Mac OSX
        echo probably run "brew install yq"
    fi

    exit 1
fi

# Specify the image type (pytorch, minigrid, or atari) as a script argument
IMAGE_TYPE=$1

# see if the user has asked for the images to be pushed as well
PUSH_IMAGES=$2
if [ "$PUSH_IMAGES" == "push" ]; then
    PUSH_IMAGES=true
else
    PUSH_IMAGES=false
fi

# Parse the values from the combined config.yaml file
USERNAME=$(yq e '.common.USERNAME' config.yaml)
IMAGE_NAME=$(yq e ".images.$IMAGE_TYPE.IMAGE_NAME" config.yaml)
TAG=$(yq e ".images.$IMAGE_TYPE.TAG" config.yaml)

FULL_IMAGE_NAME="$USERNAME/$IMAGE_NAME:$TAG"
echo "Building image $FULL_IMAGE_NAME"

# make sure buildx is installed, and alias "docker build" to "docker buildx"
docker buildx install

PLATFORMS=$(yq e ".images.$IMAGE_TYPE.PLATFORMS" config.yaml)
echo "Building for platforms: $PLATFORMS"

# Build image for multiple platforms
docker buildx build -t $FULL_IMAGE_NAME \
    --platform="$PLATFORMS" \
    -f dockerfiles/${IMAGE_TYPE}/Dockerfile . \
    --load

if [ $? -ne 0 ]; then
    exit 1
fi

# Push image if specified
if [ "$PUSH_IMAGES" == true ]; then

    echo "Pushing $FULL_IMAGE_NAME"
    docker buildx build -t $FULL_IMAGE_NAME \
        --platform="$PLATFORMS" \
        -f dockerfiles/${IMAGE_TYPE}/Dockerfile . \
        --push > /dev/null 2>rcrl_build.log &
fi

# # check if the config says to output a pytorch tagged image
NEW_TAG=""
if [ "$(yq e ".images.$IMAGE_TYPE.TAG_PYTORCH" config.yaml)" = "True" ]; then
    TORCH_VERSION=$(docker run --rm $FULL_IMAGE_NAME python -c 'import torch; print(torch.__version__, end="")')
    NEW_TAG="${TORCH_VERSION//+/_}"
fi

# check if the config says to output a python tagged image
if [ "$(yq e ".images.$IMAGE_TYPE.TAG_PYTHON" config.yaml)" = "True" ]; then
    PYTHON_VERSION=$(docker run --rm $FULL_IMAGE_NAME python -c 'import sys; print(".".join(map(str, sys.version_info[:3])), end="")')

    if [ -z "$NEW_TAG" ]; then
        NEW_TAG="${PYTHON_VERSION}"
    else
        NEW_TAG="${NEW_TAG}-python${PYTHON_VERSION}"
    fi
fi

# check if new_tag was set, if not, exit
if [ -z "$NEW_TAG" ]; then
    exit 0
fi

NEW_TAG="${NEW_TAG}-devell"
FULL_IMAGE_WITH_VERSIONS="$USERNAME/$IMAGE_NAME:$NEW_TAG"
# build image for multiple platforms, push latest to docker.io
docker buildx build -t $FULL_IMAGE_WITH_VERSIONS \
    --platform="$PLATFORMS" \
    -f dockerfiles/${IMAGE_TYPE}/Dockerfile . \
    --load

if [ $? -ne 0 ]; then
    exit 1
fi

if [ "$PUSH_IMAGES" == true ]; then
    echo "Pushing $FULL_IMAGE_WITH_VERSIONS"
    docker buildx build -t $FULL_IMAGE_WITH_VERSIONS \
        --platform="$PLATFORMS" \
        -f dockerfiles/${IMAGE_TYPE}/Dockerfile . \
        --push > /dev/null 2>rcrl_detailed_tag_build.log &

    # just incase we haven't finished the first background push
    wait
fi
echo "Done"
