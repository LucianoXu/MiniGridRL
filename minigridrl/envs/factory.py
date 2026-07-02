"""
Construct MiniGrid gym environments from an ``env`` config.
"""

from typing import Any, Callable

import gymnasium as gym
from minigrid import wrappers as mg_wrappers


def env_factory(cfg: dict[str, Any]) -> Any:

    if 'id' not in cfg:
        raise ValueError("env config must contain an 'id' field")
    
    id = cfg['id']

    if id == 'MiniGrid-Empty-5x5-v0':
        
        from minigrid.wrappers import ImgObsWrapper
        from gymnasium.vector import AsyncVectorEnv

        # n_envs values: None, int, or Literal['fn']
        n_envs = cfg.get('n_envs', None)
        disable_env_checker = cfg.get('disable_env_checker', False)

        def single_env():

            env = gym.make(id, 
                render_mode = "rgb_array", 
                disable_env_checker=disable_env_checker
            )
            env = ImgObsWrapper(env)
            return env
        
        def env_vec(n_envs: int):
            return AsyncVectorEnv(
                [single_env] * n_envs
            )

        if n_envs is None:
            return single_env()
        elif n_envs == 'fn':
            return env_vec
        elif isinstance(n_envs, int) and n_envs > 0:
            return env_vec(n_envs)
        else:
            raise ValueError("Invalid n_envs argument:", n_envs)
    
    else:
        raise ValueError("Unsupported environment ID:", id)

