"""
    Top-level entry point: dispatch a config to the right pipeline by config_type.

    from minigridrl import run
    run("configs/experiments/ppo_empty5x5.yaml")
"""

from pathlib import Path
from typing import Any

from .utils import load_yaml_config
from .envs.factory import env_factory


def run(args: dict | str | Path):
    """
    Load (if needed) and dispatch a config.
    """

    print(" >> Toplevel entry point for MiniGridRL.")

    if isinstance(args, (str, Path)):
        path = Path(args)
        print(" >> Loading config from", path)
        cfg: dict[str, Any] = load_yaml_config(path)

    else:
        cfg = args

    ctype = cfg["config_type"]

    if ctype == "env":
        return env_factory(cfg)

    # validate_config_type already guards this, but be explicit.
    raise ValueError(f"Unhandled config_type: {ctype!r}")
