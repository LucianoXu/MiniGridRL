from typing import Any

from .interface import IntrinsicReward


def intrinsic_reward_factory(cfg: dict[str, Any] | None) -> IntrinsicReward | None:
    '''
    Construct an IntrinsicReward from an ``agent.intrinsic_reward`` config block.
    Returns None when cfg is falsy (missing/empty) -> pure extrinsic training.
    '''
    if not cfg:
        return None

    if "id" not in cfg:
        raise ValueError("intrinsic_reward config must contain an 'id' field")

    id = cfg["id"]
    common = {k: v for k, v in cfg.items() if k != "id"}

    if id == "RND":
        from .rnd import RND
        return RND(**common)

    if id == "ICM":
        from .icm import ICM
        return ICM(**common)

    raise ValueError("Unsupported IntrinsicReward ID: " + str(id))
