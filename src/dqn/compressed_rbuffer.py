import functools
from typing import Any, Optional, Tuple, Union

from gymnasium import spaces
import lz4.frame
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize
import torch

from src.dqn.utils import ReplayBufferSamples


class CompressedReplayBuffer(ReplayBuffer):
    """Replay Buffer that stores transitions and compresses observations."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        compression_level: int = 1,
    ):
        """Replay buffer that uses LZ4 compression to reduce memory usage.

        This buffer stores observations in s compressed form using LZ4.
        Compression is cached to avoid recomputing for duplicate observations.

        Args:
            buffer_size: Maximum number of elements in the buffer
            observation_space: Observation space of the environment
            action_space: Action space of the environment
            device: PyTorch device to store tensors on
            n_envs: Number of parallel environments
            compression_level: LZ4 compression level (1-16, higher = better compression but slower)
        """
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs,
        )

        # Override the observations arrays with compressed storage
        self.compressed_observations = np.zeros((self.buffer_size, self.n_envs), dtype=object)
        self.compressed_next_observations = np.zeros((self.buffer_size, self.n_envs), dtype=object)
        self.compression_level = compression_level

        # Delete the original uncompressed arrays to free memory
        del self.observations
        del self.next_observations

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        """Add a new experience tuple to the buffer."""
        # Reshape needed when using multiple envs with discrete observations
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces
        action = action.reshape((self.n_envs, self.action_dim))

        # Compress observations
        for env_idx in range(self.n_envs):
            self.compressed_observations[self.pos, env_idx] = self.compress_observation(obs[env_idx])
            self.compressed_next_observations[self.pos, env_idx] = self.compress_observation(
                next_obs[env_idx],
            )

        # Store other data as normal
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """Sample a batch of experiences from the replay buffer."""
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        # Decompress observations
        observations = np.zeros((len(batch_inds), *self.obs_shape), dtype=self.observation_space.dtype)
        next_observations = np.zeros((len(batch_inds), *self.obs_shape), dtype=self.observation_space.dtype)

        for i, (batch_idx, env_idx) in enumerate(zip(batch_inds, env_indices)):
            observations[i] = self.decompress_observation(self.compressed_observations[batch_idx, env_idx])
            next_observations[i] = self.decompress_observation(self.compressed_next_observations[batch_idx, env_idx])

        # Normalize if needed
        observations = self._normalize_obs(observations, env).squeeze(1)
        next_observations = self._normalize_obs(next_observations, env).squeeze(1)
        data = (
            observations,
            self.actions[batch_inds, env_indices],
            next_observations,
            self.dones[batch_inds, env_indices].reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )

        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def to_torch(self, array: np.ndarray, copy: bool = False) -> torch.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return torch.tensor(array, device=self.device)
        return torch.as_tensor(array).to(self.device, non_blocking=True)

    @functools.lru_cache(maxsize=128)
    def _compress_bytes(self, hashable_bytes: bytes, compression_level: int) -> bytes:
        """Compress bytes using LZ4 with caching.

        This internal method handles the actual compression of observation data.
        Results are cached based on the input bytes to avoid recompressing duplicate data.

        Args:
            hashable_bytes: Raw bytes to compress
            compression_level: LZ4 compression level to use

        Returns:
            bytes: The compressed data
        """
        return lz4.frame.compress(hashable_bytes, compression_level=compression_level)

    def compress_observation(self, observation: np.ndarray) -> bytes:
        """Compress a single observation using LZ4.

        Converts the numpy array to bytes and compresses it using cached LZ4 compression.

        Args:
            observation: Numpy array containing the observation data

        Returns:
            bytes: Compressed observation data
        """
        obs_bytes = observation.tobytes()
        return self._compress_bytes(obs_bytes, self.compression_level)

    def decompress_observation(self, compressed_bytes: bytes) -> np.ndarray:
        """Decompress a single observation from its compressed form.

        Decompresses the bytes and reconstructs the numpy array with the correct shape
        and dtype.

        Args:
            compressed_bytes: The LZ4 compressed observation data

        Returns:
            np.ndarray: The decompressed observation array
        """
        obs = np.frombuffer(lz4.frame.decompress(compressed_bytes), dtype=self.observation_space.dtype).reshape(
            self.obs_shape
        )
        return obs
