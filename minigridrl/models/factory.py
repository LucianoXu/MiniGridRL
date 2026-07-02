"""
Construct MiniGrid Models
"""

from typing import Any

def model_factory(cfg: dict[str, Any]):

    if 'id' not in cfg:
        raise ValueError("env config must contain an 'id' field")
    
    id = cfg['id']

    if id == 'MLP':
        from .mlp import MLP

        model = MLP(
            d_obj = cfg['d_obj'],
            d_color = cfg['d_color'],
            d_state = cfg['d_state'],
            hidden_dims = cfg['hidden_dims']
        )

        return model

    elif id == 'ValueMLP':
        from .mlp import ValueMLP

        model = ValueMLP(
            d_obj = cfg['d_obj'],
            d_color = cfg['d_color'],
            d_state = cfg['d_state'],
            hidden_dims = cfg['hidden_dims']
        )

        return model

    else:
        raise ValueError("Unsupported Model ID:", id)

