from typing import Any

from .interface import IntrinsicReward


def intrinsic_reward_factory(cfg: dict[str, Any] | None) -> IntrinsicReward | None:
    '''Construct an IntrinsicReward from a config block, or None if cfg is falsy.'''
    if not cfg:
        return None
    raise ValueError("Unsupported IntrinsicReward id (no methods registered yet)")
