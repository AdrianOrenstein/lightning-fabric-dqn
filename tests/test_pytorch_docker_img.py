import subprocess
import pytest
import platform
import yaml

# Load the config from the YAML file
def load_config():
    with open("./config.yaml", "r") as file:
        return yaml.safe_load(file)

config = load_config()

@pytest.fixture
def get_platform():
    """Fixture to determine the correct platform based on the system architecture."""
    arch = platform.machine()
    if arch == "x86_64":
        return "linux/amd64"
    elif arch == "arm64" or arch == "aarch64":
        return "linux/arm64"
    else:
        pytest.fail(f"Unsupported architecture: {arch}")

def test_docker_pytorch_image_pull(get_platform):
    """Test if the Docker image can be pulled successfully based on the system architecture."""
    result = subprocess.run(
        ["docker", "pull", "--platform", get_platform, "adrianorenstein/pytorch:latest"], 
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert result.returncode == 0, f"Docker pull failed: {result.stderr.decode()}"

def test_run_pytorch_on_processor(get_platform):
    """Test if the Atari script runs successfully inside the Docker container."""
    result = subprocess.run(
        [
            "docker", "run", "--rm", "-it",
            "--platform", get_platform,
            "docker.io/adrianorenstein/pytorch:latest",
            "python -c 'import torch; a=torch.ones(50).cpu(); out=(a+a)*2; print(out.shape, out)'"
        ], 
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    assert result.returncode == 0, f"Running atari.py failed: {result.stderr.decode()}"
    
def has_nvidia_gpu():
    """Check if NVIDIA GPU is available by running nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "docker", "run", "--rm", "-it",
                "--gpus", "all",  # This flag enables the use of GPUs
                "--platform", get_platform,
                "docker.io/adrianorenstein/pytorch:latest",
                "nvidia-smi"  # Check if GPU is available inside the container
            ], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
        )
    except:
        return False
    
    return result.returncode == 0

@pytest.mark.skipif(not has_nvidia_gpu(), reason="NVIDIA GPU not detected")
def test_run_pytorch_on_accelerator(get_platform):
    """Test if the Atari script runs successfully inside the Docker container with GPU support."""
    result = subprocess.run(
        [
            "docker", "run", "--rm", "-it",
            "--gpus", "all",  # This flag enables the use of GPUs
            "--platform", get_platform,
            "docker.io/adrianorenstein/pytorch:latest",
            "python -c 'import torch; a=torch.ones(50).cuda(); out=(a+a)*2; print(out.shape, out)'"  # Check if GPU is available inside the container
        ], 
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    assert result.returncode == 0, f"Running nvidia-smi inside container failed: {result.stderr.decode()}"

if __name__ == "__main__":
    pytest.main()