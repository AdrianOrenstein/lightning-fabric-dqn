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

function run_coverage_report {
    coverage run --source=./src -m pytest src/ -m "not benchmark" &&
    coverage report -m
}

# Check if Docker is running
if [ "$DOCKER_RUNNING" == true ]
then
    echo "Inside docker instance"
    PYTHONPATH=/app/:$PYTHONPATH run_coverage_report
else
    echo "Starting up docker instance..."

    run_coverage_report_cmd=$(declare -f run_coverage_report); run_coverage_report_cmd+="; run_coverage_report"

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
            /bin/bash -c "${run_coverage_report_cmd}"
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
            /bin/bash -c "${run_coverage_report_cmd}"
    fi
fi
