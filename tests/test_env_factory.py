import gymnasium as gym

from minigridrl.envs import env_factory


def test_single_fn_returns_single_env_constructor():
    fn = env_factory(
        {
            "id": "MiniGrid-Empty-5x5-v0",
            "disable_env_checker": True,
            "n_envs": "single_fn",
        }
    )
    assert callable(fn)

    env = fn()
    try:
        assert isinstance(env, gym.Env)
        assert not isinstance(env, gym.vector.VectorEnv)
        obs, _ = env.reset(seed=0)
        # ImgObsWrapper -> raw 7x7x3 symbolic image (not batched)
        assert obs.shape == (7, 7, 3)
        # render_mode="rgb_array" -> a single top-down frame
        frame = env.render()
        assert frame.ndim == 3 and frame.shape[2] == 3
    finally:
        env.close()


def test_empty_env_option_uses_class_defaults():
    fn = env_factory({"id": "MiniGrid-EmptyEnv", "n_envs": "single_fn"})
    env = fn()
    try:
        # EmptyEnv default size is 8
        assert env.unwrapped.width == 8 and env.unwrapped.height == 8
        obs, _ = env.reset(seed=0)
        assert obs.shape == (7, 7, 3)  # FOV fixed regardless of grid size
    finally:
        env.close()


def test_empty_env_option_forwards_env_kwargs():
    fn = env_factory(
        {
            "id": "MiniGrid-EmptyEnv",
            "n_envs": "single_fn",
            "env_kwargs": {"size": 6, "max_steps": 50, "tile_size": 16},
        }
    )
    env = fn()
    try:
        assert env.unwrapped.width == 6 and env.unwrapped.height == 6
        assert env.unwrapped.max_steps == 50
        env.reset(seed=0)
        # tile_size forwarded -> top-down render is 6 * 16 = 96 px
        frame = env.render()
        assert frame.shape[0] == 6 * 16
    finally:
        env.close()
