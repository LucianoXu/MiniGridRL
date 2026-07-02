from abc import ABC, abstractmethod
from pathlib import Path
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter


class RLAgent(ABC):

    @staticmethod
    def tensorboard_write(
        writer: SummaryWriter,
        timesteps: int,
        **metrics: float | None,
    ):
        '''
        Generic scalar logger.

        Pass any number of `metric_name=value` keyword arguments; each is
        written to TensorBoard under its own tag at the given `timesteps`.

        A value of `None` marks a metric that is Not-Applicable to the current
        algorithm (e.g. `value_loss` / `clip_fraction` for a critic-free,
        non-PPO agent). Such metrics are skipped, since TensorBoard can only
        plot numeric scalars -- passing `None` at the call site documents the
        N/A intent without polluting the dashboard.
        '''
        for name, value in metrics.items():
            if value is None:
                continue
            writer.add_scalar(name, value, timesteps)



    @abstractmethod
    def train(
        self, 
        envs: gym.vector.VectorEnv,
        working_dir: str | Path
    ):
        ...

    @abstractmethod
    def save_local(
        self,
        pt_path: str | Path,
    ):
        ...

    @classmethod
    @abstractmethod
    def load_local(
        cls,
        pt_path: str | Path,
    ):
        ...

    @abstractmethod
    def test_reset(
        self,
    ):
        '''
        For single environment.
        '''
        ...

    @abstractmethod
    def test_step(
        self,
        obs,
    ):
        '''
        Takes the observation input and produce the action output.
        For single environment.
        '''
        ...