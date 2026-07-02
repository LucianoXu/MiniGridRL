"""
Construct MiniGrid gym environments from an ``env`` config.

Two concerns are kept separate so new environments are cheap to add:

* ``_build_single_env`` -- per-id: how to build ONE wrapped env. New envs add a
  branch here and nothing else.
* ``_dispatch`` -- shared across all envs: turn a single-env constructor into
  the requested form (single instance / constructor / vector env).
"""

from typing import Any, Callable

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from minigrid.wrappers import ImgObsWrapper


def _build_single_env(cfg: dict[str, Any]) -> Callable[[], gym.Env]:
    """
    Return a thunk that builds one ``ImgObsWrapper``-wrapped env for ``cfg``.
    Only the raw-env construction differs per id; the observation wrapping is
    shared.
    """
    id = cfg['id']
    env_kwargs = cfg.get('env_kwargs', {})

    if id in ['MiniGrid-Empty-5x5-v0', 'MiniGrid-FourRooms-v0']:
        # Registered id (size pinned to 5); goes through gym.make.
        disable_env_checker = cfg.get('disable_env_checker', False)

        def build() -> gym.Env:
            return gym.make(
                id,
                render_mode="rgb_array",
                disable_env_checker=disable_env_checker,
                **env_kwargs,
            )

    elif id == 'MiniGrid-EmptyEnv':
        from minigrid.envs.empty import EmptyEnv

        def build() -> gym.Env:
            return EmptyEnv(render_mode="rgb_array", **env_kwargs)

    else:
        raise ValueError("Unsupported environment ID:", id)

    def single_env() -> gym.Env:
        return ImgObsWrapper(build())

    return single_env


def _dispatch(single_env: Callable[[], gym.Env], n_envs: Any) -> Any:
    """
    Turn a single-env constructor into the requested form.

    n_envs values:
      * ``None``        -> a single constructed env instance
      * ``'single_fn'`` -> the single-env constructor itself (training site vectorizes)
      * ``'fn'``        -> ``Callable[[int], VectorEnv]``
      * positive ``int``-> a constructed VectorEnv of that many envs
    """
    if n_envs is None:
        return single_env()
    elif n_envs == 'single_fn':
        return single_env
    elif n_envs == 'fn':
        return lambda n: AsyncVectorEnv([single_env] * n)
    elif isinstance(n_envs, int) and n_envs > 0:
        return AsyncVectorEnv([single_env] * n_envs)
    else:
        raise ValueError("Invalid n_envs argument:", n_envs)


def env_factory(cfg: dict[str, Any]) -> Any:

    if 'id' not in cfg:
        raise ValueError("env config must contain an 'id' field")

    single_env = _build_single_env(cfg)
    return _dispatch(single_env, cfg.get('n_envs', None))
