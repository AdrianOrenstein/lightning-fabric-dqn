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

# Parse the values from the config.yaml file
USERNAME=$(yq e '.common.USERNAME' config.yaml)
IMAGE_NAME=$(yq e '.common.IMAGE_NAME' config.yaml)
TAG=$(yq e '.common.TAG' config.yaml)

FULL_IMAGE_NAME="$USERNAME/$IMAGE_NAME:$TAG"

echo "Using image: $FULL_IMAGE_NAME"

function run_linting {
    pre-commit run --all-files
}


# Check if Docker is running
if [ "$DOCKER_RUNNING" == true ]
then
    echo "Inside docker instance"
    PYTHONPATH=/app/:$PYTHONPATH run_linting

else
    echo "Starting up docker instance..."

    run_linting_cmd=$(declare -f run_linting); run_linting_cmd+="; run_linting"

    # Set volumes
    cmp_volumes="--volume=$(pwd):/app/:rw"

    # Check OS type
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # Mac OS X
        docker run --rm -it \
            --user $(id -u) \
            $cmp_volumes \
            -w /app/ \
            -e PYTHONPATH=/app/:$PYTHONPATH \
            --ipc host \
            $FULL_IMAGE_NAME \
            /bin/bash -c "${run_linting_cmd}"
    else
        # Other OS (assuming Linux)
        docker run --rm -it \
            --user $(id -u) \
            $cmp_volumes \
            -w /app/ \
            -e PYTHONPATH=/app/:$PYTHONPATH \
            --gpus all \
            --ipc host \
            $FULL_IMAGE_NAME \
            /bin/bash -c "${run_linting_cmd}"
    fi
fi
