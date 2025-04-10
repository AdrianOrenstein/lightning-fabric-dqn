import os
import torch
from typing import Optional
import time
from lightning.fabric import Fabric
from tqdm import tqdm
import wandb
from wandb.integration.lightning.fabric import WandbLogger
from dataclasses import dataclass

from src.dqn.utils import TimeScaleMeanBuffer
from src.dqn.agent import DQNAgent
from torchrl._utils import timeit

from typing import Any, Dict, NamedTuple

import numpy as np
import gymnasium as gym


@dataclass
class TrainerArgs:
    """Arguments for training setup and environment."""

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    matmul_precision: str = "high"
    """the precision of matrix multiplication in pytorch"""
    progress_bar: bool = True
    """if toggled, show tqdm progress bar"""
    env_id: str = "PongNoFrameskip-v4"
    """the id of the environment"""
    total_frames: int = 200_000_000
    """total frames of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    frame_skip: int = 5
    """the number of frames to skip at each step"""


class GymTransition(NamedTuple):
    """Transition tuple generated by a gym environment."""

    observations: np.uint8
    actions: np.uint8
    next_observations: np.uint8
    rewards: np.float32
    terminated: np.bool
    truncated: np.bool
    info: Dict[str, Any]


class BatchTrainer:
    ARGS = TrainerArgs

    def __init__(
        self,
        args: TrainerArgs,
        fabric: Fabric,
        logger: WandbLogger,
        envs: gym.vector.SyncVectorEnv,
        agent: DQNAgent,
    ):
        """Initialize the batch trainer.

        Args:
            args: Combined training and agent arguments
            fabric: Pre-configured Fabric instance
            logger: Pre-configured WandbLogger instance
        """
        self.args: TrainerArgs = args
        self.fabric: Fabric = fabric
        self.logger: WandbLogger = logger

        assert args.num_envs == 1, "vectorized envs are not supported at the moment"

        # Setup directories
        self.data_path = logger.experiment.dir
        if args.save_model:
            os.makedirs(f"{self.data_path}/weights", exist_ok=True)
        if args.capture_video:
            os.makedirs(f"{self.data_path}/videos", exist_ok=True)

        # Seeding
        fabric.seed_everything(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        # Setup environment and agent
        self.envs = envs
        self.agent = agent

        # Initialize metrics
        self.start_time = time.time()
        self.step_count = 0
        self.e_count = 0
        self.frame_no = 0
        self.ten_logged_frame = 0  # ten thousand frames
        self.hun_logged_frame = 0  # hundred thousand frames
        self.mil_logged_frame = 0  # million frames

        # Initialize metric buffers
        self.episode_reward = TimeScaleMeanBuffer(100)
        self.episodic_length = TimeScaleMeanBuffer(100)
        self.episode_time = TimeScaleMeanBuffer(100)
        self.per_episode_decision_count = 0
        self.per_episode_num_decisions_taken = TimeScaleMeanBuffer(100)
        self.normalised_reward = TimeScaleMeanBuffer(
            60 * 10, additional_timescales=[second * 60 for second in [1, 3, 5, 10]]
        )

        self.agent_keys = ["Replay Buffer Add", "Agent Action", "Replay Buffer Sample", "Update"]
        self.environment_keys = ["Environment Step"]

    def execute_agent_decision(self, action) -> tuple[np.ndarray, np.ndarray, bool, bool, dict]:
        """Execute an action in the environment.

        Args:
            action: The action to execute

        Returns:
            Tuple of (next_obs, reward, termination, truncation, info)
        """
        with timeit("Environment Step"):
            next_obs, reward, termination, truncation, info = self.envs.step(action)

        self.step_count += 1
        self.frame_no = int(info["frame_number"][0])

        # Track normalized reward
        self.normalised_reward.add(reward[0])

        return next_obs, reward, termination, truncation, info

    def collect_rollouts(self) -> None:
        """Collect experience from the environment."""
        for _ in range(self.agent.args.train_frequency):
            with timeit("Agent Action"):
                epsilon = self.agent.get_exploration_rate(self.frame_no)
                if np.random.random() < epsilon:
                    action = self.envs.action_space.sample()
                else:
                    with timeit("Agent Decision"), torch.no_grad(), self.fabric.autocast():
                        obs = torch.as_tensor(self.obs).to(self.fabric.device)
                        action = self.agent.policy(obs, self.agent.q_network)
                        self.agent.agent_decisions_made_so_far += 1
                        self.per_episode_decision_count += 1
                    action = action.cpu().numpy()

            next_obs, reward, termination, truncation, info = self.execute_agent_decision(action)

            with timeit("Replay Buffer Add"):
                self.agent.store_experience(self.obs, next_obs, action, reward, termination, [])

                if "episode" in info:
                    self.e_count += 1
                    self.episode_reward.add(info["episode"]["r"][0])
                    self.episodic_length.add(info["episode"]["l"][0])
                    self.episode_time.add(info["episode"]["t"][0])
                    self.per_episode_num_decisions_taken.add(self.per_episode_decision_count)

                    self.per_episode_decision_count = 0

            self.obs = next_obs

    def prefill_buffer(self, pbar: Optional[tqdm] = None) -> None:
        """Prefill the replay buffer with random actions."""
        for i in range(self.agent.args.learning_starts):
            action = self.envs.action_space.sample()

            next_obs, reward, termination, truncation, info = self.execute_agent_decision(action)

            with timeit("Replay Buffer Add"):
                self.agent.store_experience(self.obs, next_obs, action, reward, termination, [])

                if "episode" in info:
                    self.e_count += 1
                    self.episode_reward.add(info["episode"]["r"][0])
                    self.episodic_length.add(info["episode"]["l"][0])
                    self.episode_time.add(info["episode"]["t"][0])

            self.obs = next_obs

            if pbar and self.args.progress_bar and i % 1000 == 0:
                pbar.n = self.frame_no
                pbar.desc = f"{'frames': <8}: {self.frame_no} | {'agent steps': <8}: {self.step_count}"
                pbar.refresh()

    def train(self) -> None:
        """Main training loop."""
        # Setup progress bar
        if self.args.progress_bar:
            self.pbar = tqdm(total=self.args.total_frames)
        else:
            self.pbar = None

        self.obs, starting_info = self.envs.reset(seed=self.args.seed)
        self.frame_no = int(starting_info["frame_number"][0])
        del starting_info

        # Prefill replay buffer
        with timeit("Prefill Replay Buffer"):
            self.prefill_buffer(self.pbar)

        # Start training
        while self.frame_no < self.args.total_frames:
            # Collect experience
            with timeit("Agent Rollout"):
                self.collect_rollouts()

            with timeit("Replay Buffer Sample"), self.fabric.autocast():
                data = self.agent.replay_buffer.sample(self.agent.args.batch_size)

            with timeit("Update"), self.fabric.autocast():
                loss, est_q_values = self.agent.update(data=data)
                loss = loss.item()
                est_q_values = est_q_values.mean().item()

            # Progress bar and logging
            stepped_frame = self.frame_no // 10_000
            if self.pbar and self.args.progress_bar and stepped_frame > self.ten_logged_frame:
                self.ten_logged_frame = stepped_frame
                stepped_frame = stepped_frame * 10_000
                self.pbar.n = stepped_frame
                self.pbar.desc = f"{'frames': <8}: {stepped_frame} | {'agent steps': <8}: {self.step_count} | {'agent decisions': <8}: {self.agent.agent_decisions_made_so_far} | {'agent updates': <8}: {self.agent.update_count}"
                self.pbar.refresh()

            stepped_frame = self.frame_no // 100_000
            if stepped_frame > self.hun_logged_frame:
                self.hun_logged_frame = stepped_frame
                stepped_frame = stepped_frame * 100_000
                with timeit("Logging"):
                    self.log_metrics(loss, est_q_values, stepped_frame)

            if self.args.save_model:
                stepped_frame = self.frame_no // 1_000_000
                if stepped_frame > self.mil_logged_frame:
                    self.mil_logged_frame = stepped_frame
                    self.save_checkpoint(stepped_frame)

        self.envs.close()
        wandb.finish()

    def log_metrics(self, loss: float, est_q_values: float, stepped_frame: int) -> None:
        """Log metrics to the logger."""
        elapsed_time = time.time() - self.start_time
        timeit_percall = timeit.todict(prefix="Timeit_percall")
        timeit_accumulative = timeit.todict(percall=False, prefix="Timeit_acculumative")

        agent_time_percall = sum(timeit_percall["Timeit_percall/" + name] for name in self.agent_keys)
        environment_time_percall = sum(timeit_percall["Timeit_percall/" + name] for name in self.environment_keys)
        agent_time_accumulative = sum(timeit_accumulative["Timeit_acculumative/" + name] for name in self.agent_keys)
        environment_time_accumulative = sum(
            timeit_accumulative["Timeit_acculumative/" + name] for name in self.environment_keys
        )

        agent_environment_ratio = agent_time_percall / environment_time_percall
        agent_environment_ratio_accumulative = agent_time_accumulative / environment_time_accumulative

        to_log = {
            "episode/episode_reward": self.episode_reward.mean(),
            "episode/episodic_length": self.episodic_length.mean(),
            "episode/episode_time": self.episode_time.mean(),
            "charts/td_loss": loss,
            "charts/avg_q_values": est_q_values,
            "speed_per_sec/FPS": stepped_frame // elapsed_time,
            "speed_per_sec/SPS": self.step_count // elapsed_time,
            "speed_per_sec/UPS": self.agent.update_count // elapsed_time,
            "speed_per_frame/UPF": self.agent.update_count / stepped_frame,
            "speed_per_frame/SPF": self.step_count / stepped_frame,
            **timeit_percall,
            **timeit_accumulative,
            "summary_percall/agent_time": agent_time_percall,
            "summary_percall/environment_time": environment_time_percall,
            "summary_percall/agent_environment_ratio": agent_environment_ratio,
            "summary_accumulative/agent_time": agent_time_accumulative,
            "summary_accumulative/environment_time": environment_time_accumulative,
            "summary_accumulative/agent_environment_ratio": agent_environment_ratio_accumulative,
            "xaxis/env_frames": stepped_frame,
            "xaxis/env_steps": self.step_count,
            "xaxis/agent_decisions": self.agent.agent_decisions_made_so_far,
            "xaxis/agent_updates": self.agent.update_count,
            "xaxis/agent_target_update": self.agent.target_update_count,
            "charts/rbuffer_size": self.agent.replay_buffer.size(),
            "agent/per_episode_num_decisions_taken": self.per_episode_num_decisions_taken.mean(),
            **{
                f"reward_rates/{second}sec": self.normalised_reward.mean(timescale=second * 60)
                for second in [1, 3, 5, 10]
            },
        }
        self.fabric.log_dict(to_log, step=stepped_frame)

    def save_checkpoint(self, stepped_frame: int) -> None:
        """Save a model checkpoint."""
        model_path = f"{self.data_path}/weights/{self.args.exp_name}-{stepped_frame}M.dqn_model"
        self.agent.save(model_path)
        print(f"model saved to {model_path}")
