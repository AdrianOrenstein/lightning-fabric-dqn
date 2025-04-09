import os
import subprocess
import pytest


def test_pull_docker_atari_pytorch():
    command = [
        "docker", "pull", 
        "--platform", "linux/amd64",
        "adrianorenstein/atari_pytorch:latest",
    ]

    result = subprocess.run(' '.join(command), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    assert result.returncode == 0, f"Command failed: {result.stderr.decode('utf-8')}"
    

def test_docker_atari_pytorch():
    command = [
        "docker", "run", "--rm", "-t",
        "-v", f"{os.getcwd()}:/app",
        "-w", "/app",
        "--platform", "linux/amd64",
        "adrianorenstein/atari_pytorch:latest",
        "python", "-c",
        "\"import gymnasium as gym; env = gym.make('ALE/Breakout-v5'); env.reset(); env.step(env.action_space.sample()); env.close()\""
    ]

    result = subprocess.run(' '.join(command), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    assert result.stdout.decode('utf-8').strip().endswith("[Powered by Stella]"), f"Command failed: {result.stderr.decode('utf-8')}"
    
if __name__ == "__main__":
    pytest.main()
    