BootStrap: docker
From: adrianorenstein/pytorch:latest

%post
    # Force reinstall and recompile all pip packages
    pip freeze | xargs pip install --force-reinstall --ignore-installed --no-binary :all: