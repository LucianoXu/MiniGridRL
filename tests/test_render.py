import tempfile

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from minigridrl.agents.REINFORCE import REINFORCE
from minigridrl.envs import env_factory
from minigridrl.models.mlp import MLP
from minigridrl.render import (
    _BG_COLOR,
    _DIVIDER_WIDTH,
    compose_frame,
    decode_ego_frame,
    frames_to_video_tensor,
    record_rollout_video,
)


def _read_rollout_images(logdir, tag="rollout"):
    ea = EventAccumulator(logdir, size_guidance={"images": 0})
    ea.Reload()
    if tag not in ea.Tags().get("images", []):
        return []
    return ea.Images(tag)


def test_frames_to_video_tensor_shape_and_dtype():
    # 5 RGB frames of size 12x20
    frames = [np.zeros((12, 20, 3), dtype=np.uint8) for _ in range(5)]

    video = frames_to_video_tensor(frames)

    # add_video expects (N, T, C, H, W)
    assert video.shape == (1, 5, 3, 12, 20)
    assert video.dtype == torch.uint8


def test_decode_ego_frame_renders_valid_observation():
    # A valid MiniGrid 7x7x3 observation: all "empty" cells (object idx 1,
    # color 0, state 0) -- decodes without raising.
    obs = np.zeros((7, 7, 3), dtype=np.uint8)
    obs[..., 0] = 1  # OBJECT_TO_IDX["empty"]

    frame = decode_ego_frame(obs)

    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.dtype == np.uint8
    # 7x7 grid at 32 px/tile -> 224x224
    assert frame.shape[0] == 7 * 32 and frame.shape[1] == 7 * 32


def test_compose_frame_places_panels_side_by_side():
    top_down = np.zeros((160, 160, 3), dtype=np.uint8)
    ego = np.zeros((224, 224, 3), dtype=np.uint8)

    frame = compose_frame(top_down, ego, label="t=1234")

    assert frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[2] == 3
    # panels scaled to a common height (the taller of the two)
    assert frame.shape[0] == 224
    # horizontally concatenated: wider than either panel alone
    assert frame.shape[1] > 224


def test_compose_frame_centers_smaller_panel_without_upscaling():
    # State much larger than the (fixed 7x7) observation.
    state = np.full((800, 800, 3), (0, 180, 0), dtype=np.uint8)  # green
    ego = np.full((224, 224, 3), (200, 0, 0), dtype=np.uint8)    # red

    frame = compose_frame(state, ego, label="t=0")

    # both panels share the same box size (the larger of the two), side by side
    assert frame.shape[0] == 800
    assert frame.shape[1] == 800 + _DIVIDER_WIDTH + 800

    right = frame[:, 800 + _DIVIDER_WIDTH:]
    assert right.shape == (800, 800, 3)

    # observation is NOT upscaled: exactly its native pixel count remains
    red = np.all(right == (200, 0, 0), axis=-1)
    assert red.sum() == 224 * 224
    # centered -> panel corner is the padding background, distinct from the
    # observation and from MiniGrid's black "unseen" tiles
    assert tuple(right[0, 0]) == _BG_COLOR
    assert _BG_COLOR != (0, 0, 0)
    # center of the right panel falls inside the observation
    assert tuple(right[400, 400]) == (200, 0, 0)


def test_record_rollout_video_smoke():
    agent = REINFORCE(policy=MLP())
    agent.policy.train()  # start in train mode
    env_fn = env_factory(
        {
            "id": "MiniGrid-Empty-5x5-v0",
            "disable_env_checker": True,
            "n_envs": "single_fn",
        }
    )
    d = tempfile.mkdtemp()
    writer = SummaryWriter(d)

    record_rollout_video(
        agent, env_fn, writer, timesteps=1000, seed=0, fps=8, max_steps=5
    )
    writer.flush()
    writer.close()

    imgs = _read_rollout_images(d)
    assert len(imgs) == 1
    assert imgs[0].step == 1000
    assert imgs[0].encoded_image_string[:6] == b"GIF89a"
    # recording must not leave the policy stuck in eval mode
    assert agent.policy.training is True
