from ..envs import env_factory
from ..agents import RLAgent, agent_factory
import gymnasium as gym
from pathlib import Path

def rl_train(
    cfg: dict,
    working_dir: Path
):
    envs_cfg = cfg['envs']
    envs = env_factory(envs_cfg)

    # check whether result is VectorEnv
    if not isinstance(envs, gym.vector.VectorEnv):
        raise ValueError("The constructed environment must be VectorEnv, but got", type(envs))

    agent_cfg = cfg['agent']
    agent: RLAgent = agent_factory(agent_cfg)

    rl_training_dir = Path(cfg['rl_training_dir'])
    training_cfg = cfg['training_cfg']
    agent.train(
        envs,
        working_dir= working_dir / rl_training_dir,
        **training_cfg
    )