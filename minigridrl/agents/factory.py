"""
Construct MiniGrid Models
"""

from typing import Any
from .interface import RLAgent
from ..models import model_factory

def agent_factory(cfg: dict[str, Any]) -> RLAgent:

    if 'id' not in cfg:
        raise ValueError("env config must contain an 'id' field")
    
    id = cfg['id']

    if id == 'REINFORCE':
        from .REINFORCE import REINFORCE

        model_cfg = cfg['model']
        model = model_factory(model_cfg)

        agent = REINFORCE(
            policy = model,
            lr = cfg['lr'],
            betas = tuple(cfg['betas']),
            eps = cfg['eps'],
            weight_decay = cfg['weight_decay'],
            grad_norm_clip = cfg['grad_norm_clip']
        )

        return agent

    elif id == 'PPO':
        from .PPO import PPO

        policy = model_factory(cfg['model'])
        value_net = model_factory(cfg['value_model'])

        agent = PPO(
            policy = policy,
            value_net = value_net,
            lr = cfg['lr'],
            betas = tuple(cfg['betas']),
            eps = cfg['eps'],
            weight_decay = cfg['weight_decay'],
            grad_norm_clip = cfg['grad_norm_clip'],
            gamma = cfg['gamma'],
            gae_lambda = cfg['gae_lambda'],
            clip_range = cfg['clip_range'],
            clip_range_vf = cfg.get('clip_range_vf', None),
            n_epochs = cfg['n_epochs'],
            num_minibatches = cfg['num_minibatches'],
            vf_coef = cfg['vf_coef'],
            ent_coef = cfg['ent_coef'],
            normalize_advantage = cfg.get('normalize_advantage', True),
            target_kl = cfg.get('target_kl', None),
        )

        return agent

    else:
        raise ValueError("Unsupported Agent ID:", id)

