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
    
    else:
        raise ValueError("Unsupported Agent ID:", id)

