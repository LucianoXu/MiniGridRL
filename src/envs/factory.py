"""
Construct MiniGrid gym environments from an ``env`` config.
"""

from typing import Any, Callable

import gymnasium as gym
from minigrid import wrappers as mg_wrappers


def env_factory(cfg: dict[str, Any]):

    if 'id' not in cfg:
        raise ValueError("env config must contain an 'id' field")
    
    id = cfg['id']

    if id == 'MiniGrid-Empty-5x5-v0':
        
        from minigrid.wrappers import ImgObsWrapper
        from gymnasium.wrappers import FlattenObservation
        from gymnasium.vector import AsyncVectorEnv

        n_envs = cfg.get('n_envs', 1)
        disable_env_checker = cfg.get('disable_env_checker', False)

        def single_env():

            env = gym.make(id, 
                render_mode = "rgb_array", 
                disable_env_checker=disable_env_checker
            )
            env = FlattenObservation(ImgObsWrapper(env))
            return env
        
        env = AsyncVectorEnv(
            [single_env] * n_envs
        )

        return env
    
    else:
        raise ValueError("Unsupported environment ID:", id)

