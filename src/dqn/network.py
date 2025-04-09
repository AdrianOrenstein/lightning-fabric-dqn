import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LazyLayerNorm(nn.Module):
    def __init__(self):
        """Lazy Layer Norm remembers the input size instead of needing it pre-defined"""

        super().__init__()

    def forward(self, input):
        """Layer Norm"""
        return F.layer_norm(input, input.size())

    def extra_repr(self) -> str:
        return "Layer Normalization"


class LazyRMSNorm(nn.Module):
    """Lazy RMS Normalization remembers the input size instead of needing it pre-defined"""

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return F.rms_norm(input, input.size())

    def extra_repr(self) -> str:
        return "RMS Normalization"


class QNetwork(nn.Module):
    """Q-value network for estimating state-action values."""

    def __init__(self, env: gym.vector.SyncVectorEnv, action_space_size: int | None = None):
        """Initialize the Q-value network."""
        super().__init__()

        # Note: bias = false if we use normalisation.
        # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm

        conv_layers = [
            nn.LazyConv2d(32, 8, stride=4, bias=False),
            LazyRMSNorm(),
            nn.ReLU(),
            nn.LazyConv2d(64, 4, stride=2, bias=False),
            LazyRMSNorm(),
            nn.ReLU(),
            nn.LazyConv2d(64, 3, stride=1, bias=False),
            LazyRMSNorm(),
            nn.ReLU(),
        ]

        linear_layers = [
            nn.Flatten(),
            nn.LazyLinear(512, bias=False),
            LazyRMSNorm(),
            nn.ReLU(),
            nn.LazyLinear(action_space_size or env.single_action_space.n),  # type: ignore
        ]

        self.network = nn.Sequential(*conv_layers, *linear_layers)

        # initialise the conv_layers with layer_init
        if env.observation_space.shape is not None:
            self.network(torch.randn(size=env.observation_space.shape))
        else:
            self.network(torch.randn(size=(1, 4, 84, 84)))

        for module in self.network.modules():
            if isinstance(module, nn.Conv2d):
                torch.nn.init.orthogonal_(module.weight, np.sqrt(2))
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight, nonlinearity="leaky_relu")
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x / 255.0)
