import torch
from gymnasium.vector import AutoresetMode

from minigridrl.agents import RLAgent
from minigridrl.envs import env_factory


def test_vectorize_builds_next_step_vector_env():
    env_fn = env_factory(
        {
            "id": "MiniGrid-Empty-5x5-v0",
            "disable_env_checker": True,
            "n_envs": "single_fn",
        }
    )

    venv = RLAgent.vectorize(env_fn, 4)
    try:
        assert venv.num_envs == 4
        # training requires NEXT_STEP autoreset semantics
        assert venv.metadata["autoreset_mode"] == AutoresetMode.NEXT_STEP
        # NumpyToTorch wrapper -> batched torch observations
        obs, _ = venv.reset(seed=0)
        assert isinstance(obs, torch.Tensor)
        assert tuple(obs.shape) == (4, 7, 7, 3)
    finally:
        venv.close()
