import collections
from typing import Any, Dict, List, NamedTuple, Optional
import numpy as np
import torch


class GymTransition(NamedTuple):
    """Transition tuple generated by a gym environment."""

    observations: np.uint8
    actions: np.uint8
    next_observations: np.uint8
    rewards: np.float32
    terminated: np.bool
    truncated: np.bool
    info: Dict[str, Any]


class ReplayBufferSamples(NamedTuple):
    """Samples from a replay buffer."""

    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor


class TimeScaleMeanBuffer:
    """Stores a deque of items and calculates the mean over different timescales.
    Example use
    normalised_reward = TimeScaleMeanBuffer()
    for second in [1, 3, 5, 10]:
        to_log[f"reward_rates/{second}sec"] = normalised_reward.mean(timescale=second * 60)
    """

    def __init__(self, capacity: int, additional_timescales: Optional[List[int]] = []) -> None:
        self.capacity: int = capacity
        self.timescales: List[int] = [capacity] + additional_timescales
        assert all(timescale <= self.capacity for timescale in self.timescales), "All timescales must be <= capacity"
        self.deque: List[int] = collections.deque(maxlen=capacity)

    def add(self, val: float) -> None:
        """Add to the buffer with the current timestamp."""
        self.deque.append(val)

    def mean(self, timescale: Optional[int] = None) -> float:
        """
        Retrieve the mean.

        If timescale is None, return the mean of all values.
        Otherwise, return the mean of values within the specified timescale.
        """
        if not self.deque:
            return None

        if timescale is None:
            timescale = self.capacity

        if timescale not in self.timescales:
            raise ValueError("timescale not recorded.")

        actual_timescale = min(timescale, len(self.deque))
        values = list(self.deque)[-actual_timescale:]
        return sum(values) / actual_timescale
