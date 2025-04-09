import numpy as np
from typing import Optional
from lightning.fabric import Fabric
from wandb.integration.lightning.fabric import WandbLogger
from torchrl._utils import timeit
from dataclasses import dataclass

from src.dqn.batch_trainer import BatchTrainer, TrainerArgs
from src.option_dqn.option_agent import OptionDQNAgent


@dataclass
class BatchedOptionTrainerArgs(TrainerArgs):
    """Arguments for temporal option training."""

    option_deliberation_cost: Optional[float] = 0
    """Cost added to reward when agent makes a new decision"""


class OptionDQNBatchTrainer(BatchTrainer):
    """Trainer class for temporal DQN with action repeats."""

    ARGS = BatchedOptionTrainerArgs

    def __init__(
        self,
        args: BatchedOptionTrainerArgs,
        fabric: Fabric,
        logger: WandbLogger,
        envs,
        agent: OptionDQNAgent,
    ):
        super().__init__(args, fabric, logger, envs, agent)
        self.args: BatchedOptionTrainerArgs = args
        self.agent: OptionDQNAgent = agent

    def execute_agent_decision(self, action) -> tuple[np.ndarray, np.ndarray, bool, bool, dict]:
        """Execute an option (repeated action) in the environment.

        Args:
            action: The action to execute

        Returns:
            Tuple of (next_obs, accumulated_reward, termination, truncation, info)
        """
        base_action, option_length = self.agent.decode_action(action[0])
        assert option_length > 0, "Option length must be greater than 0"

        # Store initial observation for experience
        accumulated_reward = 0

        # Apply computation cost if specified
        if self.args.option_deliberation_cost is not None:
            accumulated_reward = self.args.option_deliberation_cost

        # Execute action for option_length steps
        for repeat_i in range(option_length):
            with timeit("Environment Step"):
                next_obs, reward, termination, truncation, info = self.envs.step(np.array([base_action]))
                self.step_count += 1
                self.frame_no = int(info["frame_number"][0])

                # Accumulate discounted reward
                accumulated_reward += (self.agent.args.gamma**repeat_i) * reward

                if termination or truncation:
                    break

        # Track normalized reward. I've chosen to not normalize the reward during option training.
        self.normalised_reward.add(accumulated_reward[0])

        return next_obs, accumulated_reward, termination, truncation, info
