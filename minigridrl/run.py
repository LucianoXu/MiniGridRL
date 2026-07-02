"""
    Top-level entry point: dispatch a config to the right pipeline by config_type.

    from minigridrl import run
    run("configs/experiments/ppo_empty5x5.yaml")
"""
import shutil
from pathlib import Path
from typing import Any

from .utils import load_yaml_config, save_yaml_config, tee_console
from .envs import env_factory
from .models import model_factory
from .agents import agent_factory
from .experiments.rl_train import rl_train

def run(args: dict | str | Path, overwrite = False):
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
    elif ctype == "model":
        return model_factory(cfg)
    elif ctype == "agent_factory":
        return agent_factory(cfg)
    

    else:
        # the range of experiments

        # all subsequent paths will be considered relative to the working_dir
        working_dir = Path(cfg['working_dir'])
        print(" >> Working directory:", working_dir)

        if working_dir.exists():

            if overwrite:
                print(" >> Cleaning up existing working directory.")
                shutil.rmtree(working_dir)
            else:
                raise FileExistsError(
                    f"Working directory already exists: {working_dir}. "
                    f"Use overwrite=True to remove it."
                )            

        working_dir.mkdir(parents=True, exist_ok=False)

        # output yaml to the working dir as a record
        save_yaml_config(cfg, working_dir / "config.yaml")

        # mirror everything printed to the console into the working dir so each
        # experiment folder keeps a full run log.
        with tee_console(working_dir / "console.log"):
            if ctype == "rl_train":
                result = rl_train(cfg, working_dir=working_dir)
            else:
                # validate_config_type already guards this, but be explicit.
                raise ValueError(f"Unhandled config_type: {ctype!r}")

            # forge the finish remark
            with open(working_dir / "DONE", "w") as p:
                p.write("THIS TASK COMPLETED GRACEFULLY.")

        return result
