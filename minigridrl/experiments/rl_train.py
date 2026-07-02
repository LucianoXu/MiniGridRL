from ..envs import env_factory
from ..agents import RLAgent, agent_factory
from pathlib import Path

def rl_train(
    cfg: dict,
    working_dir: Path
):
    # env config describes ONE env; the agent vectorizes it using n_envs from
    # training_cfg. Request the single-env constructor here.
    envs_cfg = cfg['envs']
    env_fn = env_factory({**envs_cfg, 'n_envs': 'single_fn'})

    agent_cfg = cfg['agent']
    agent: RLAgent = agent_factory(agent_cfg)

    rl_training_dir = Path(cfg['rl_training_dir'])
    training_cfg = cfg['training_cfg']
    agent.train(
        env_fn,
        working_dir= working_dir / rl_training_dir,
        **training_cfg
    )