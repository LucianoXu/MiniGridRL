"""
Periodic roll-out video recording for TensorBoard.

Runs a single stochastic evaluation episode on a fresh single environment and
records a two-panel video -- left: the full top-down environment state; right:
the agent's egocentric 7x7x3 observation rendered as MiniGrid sprites -- under
the ``rollout`` tag via ``SummaryWriter.add_video``.
"""

from __future__ import annotations

import os

# TensorBoard's `add_video` imports `moviepy.editor`, which imports
# `moviepy.video.io.preview` -- a module whose body runs `pygame.init()`. That
# initializes pygame's *video* subsystem with a real, windowed SDL driver
# (e.g. "cocoa" on macOS), which spawns an app in the Dock and can steal focus,
# even though we only ever encode a GIF and never show a window. Forcing SDL's
# headless "dummy" driver before that import chain runs keeps video recording
# side-effect free. `setdefault` leaves an explicit user override untouched.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from PIL import Image, ImageDraw

from minigrid.core.grid import Grid
from minigrid.core.constants import TILE_PIXELS

# Width of the vertical separator between the two panels, in pixels.
_DIVIDER_WIDTH = 4
_DIVIDER_COLOR = (60, 60, 60)
_LABEL_COLOR = (255, 255, 255)
# Padding background behind a centered (smaller) panel. Deliberately distinct
# from MiniGrid's black "unseen" tiles so the real image region is tellable
# apart from the composite's background.
_BG_COLOR = (30, 30, 40)
# 1-px outline drawn around a centered panel's content, in the padding.
_CONTENT_BORDER_COLOR = (200, 200, 80)

# Standard egocentric pose: the agent sits at the bottom-center of its 7x7
# field of view, facing up (direction 3). Matches how MiniGrid lays out the
# partial observation the policy receives.
_EGO_AGENT_POS = (3, 6)
_EGO_AGENT_DIR = 3


def decode_ego_frame(obs_img: np.ndarray) -> np.ndarray:
    """
    Render the agent's egocentric 7x7x3 symbolic observation as an RGB image of
    MiniGrid sprites, in the standard partial-view pose (agent bottom-center,
    facing up). Returns a ``(7*TILE_PIXELS, 7*TILE_PIXELS, 3)`` uint8 array.
    """
    grid, vis_mask = Grid.decode(np.asarray(obs_img, dtype=np.uint8))
    return grid.render(
        TILE_PIXELS,
        agent_pos=_EGO_AGENT_POS,
        agent_dir=_EGO_AGENT_DIR,
        highlight_mask=vis_mask,
    )


def _place_centered(content: np.ndarray, box_h: int, box_w: int) -> np.ndarray:
    """
    Center ``content`` (unscaled) on a ``(box_h, box_w)`` background canvas. If
    the content is smaller than the box, outline its region so the real image
    is distinguishable from the padding background.
    """
    ch, cw = content.shape[:2]
    canvas = np.full((box_h, box_w, 3), _BG_COLOR, dtype=np.uint8)
    top = (box_h - ch) // 2
    left = (box_w - cw) // 2
    canvas[top:top + ch, left:left + cw] = content

    if ch < box_h or cw < box_w:
        # 1-px rectangle drawn in the padding, just outside the content.
        y0, y1 = max(0, top - 1), min(box_h - 1, top + ch)
        x0, x1 = max(0, left - 1), min(box_w - 1, left + cw)
        canvas[y0, x0:x1 + 1] = _CONTENT_BORDER_COLOR
        canvas[y1, x0:x1 + 1] = _CONTENT_BORDER_COLOR
        canvas[y0:y1 + 1, x0] = _CONTENT_BORDER_COLOR
        canvas[y0:y1 + 1, x1] = _CONTENT_BORDER_COLOR

    return canvas


def compose_frame(top_down: np.ndarray, ego: np.ndarray, label: str) -> np.ndarray:
    """
    Build a single video frame from the top-down state panel (left) and the
    egocentric observation panel (right).

    Both panels share the same box size -- the larger of the two, element-wise
    -- so neither image is scaled. The smaller panel is centered on a distinct
    background with its content outlined, so its real extent is tellable apart
    from the padding. Panels are joined by a thin divider and annotated with
    ``label`` in the top-left.
    """
    top_down = np.asarray(top_down, dtype=np.uint8)
    ego = np.asarray(ego, dtype=np.uint8)

    box_h = max(top_down.shape[0], ego.shape[0])
    box_w = max(top_down.shape[1], ego.shape[1])

    left = _place_centered(top_down, box_h, box_w)
    right = _place_centered(ego, box_h, box_w)

    divider = np.full((box_h, _DIVIDER_WIDTH, 3), _DIVIDER_COLOR, dtype=np.uint8)
    frame = np.concatenate([left, divider, right], axis=1)

    canvas = Image.fromarray(frame)
    ImageDraw.Draw(canvas).text((4, 2), label, fill=_LABEL_COLOR)
    return np.asarray(canvas)


def frames_to_video_tensor(frames: list[np.ndarray]) -> torch.Tensor:
    """
    Stack a list of ``(H, W, 3)`` uint8 RGB frames into the ``(N, T, C, H, W)``
    uint8 tensor expected by ``SummaryWriter.add_video`` (single video, N=1).
    """
    stacked = np.stack(frames, axis=0)              # (T, H, W, C)
    tensor = torch.from_numpy(stacked).to(torch.uint8)
    tensor = tensor.permute(0, 3, 1, 2)             # (T, C, H, W)
    return tensor.unsqueeze(0)                      # (1, T, C, H, W)


def record_rollout_video(
    agent,
    env_fn: Callable[[], gym.Env],
    writer,
    timesteps: int,
    *,
    seed: int,
    fps: int = 8,
    max_steps: int | None = None,
) -> None:
    """
    Run one stochastic evaluation episode on a fresh single env and log a
    two-panel rollout video to ``writer`` under the ``rollout`` tag at
    ``timesteps``.

    Uses the agent's ``test_reset`` / ``test_step`` (the single-env
    demonstration interface). The policy's train/eval mode is saved and
    restored, so this is side-effect free for the surrounding training loop.
    """
    was_training = agent.policy.training
    env = env_fn()
    try:
        agent.test_reset()
        obs, _ = env.reset(seed=seed)

        frames: list[np.ndarray] = []
        step = 0
        while True:
            frames.append(
                compose_frame(env.render(), decode_ego_frame(obs), f"t={timesteps}")
            )
            act = agent.test_step(torch.as_tensor(np.asarray(obs)))
            obs, _, terminated, truncated, _ = env.step(int(act))
            step += 1
            if terminated or truncated or (max_steps is not None and step >= max_steps):
                break

        # final observation frame
        frames.append(
            compose_frame(env.render(), decode_ego_frame(obs), f"t={timesteps}")
        )
    finally:
        env.close()
        if was_training:
            agent.policy.train()

    video = frames_to_video_tensor(frames)
    writer.add_video("rollout", video, global_step=timesteps, fps=fps)
