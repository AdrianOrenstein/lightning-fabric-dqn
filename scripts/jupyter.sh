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

function run_jupyter {
    jupyter-lab --ip 0.0.0.0 --port 8080 --no-browser --allow-root
}
# Check if Docker is running
if [ "$DOCKER_RUNNING" == true ]
then
    echo "Inside docker instance"
    PYTHONPATH=/app/:$PYTHONPATH run_jupyter
else
    echo "Starting up docker instance..."

    run_jupyter_cmd=$(declare -f run_jupyter); run_jupyter_cmd+="; run_jupyter"

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
            -p 8080:8080 \
            -d $FULL_IMAGE_NAME \
            /bin/bash -c "$run_jupyter_cmd"
    else
        # Other OS (assuming Linux)
        docker run --rm -it \
            --user $(id -u) \
            $cmp_volumes \
            -w /app/ \
            -e PYTHONPATH=/app/:$PYTHONPATH \
            --ipc host \
            -p 8080:8080 \
            --name jupyter \
            -d $FULL_IMAGE_NAME \
            /bin/bash -c "$run_jupyter_cmd"
    fi
fi
