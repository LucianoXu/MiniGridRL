from typing import Sequence
import copy
import torch
from torch import nn
from torch.nn import Embedding

from .interface import RLModule, GridEmbedding, NUM_ACTIONS

class MLP(RLModule):
    def __init__(self, d_obj: 
                 int = 8, 
                 d_color: int = 4, 
                 d_state: int = 2, 
                 hidden_dims: Sequence[int] = (96,)):
        super().__init__()

        self.hidden_dims = copy.copy(hidden_dims)
        self.embedding = GridEmbedding(d_obj, d_color, d_state)

        current_dim = self.embedding.entire_embedding_dim

        layers = []

        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(current_dim, hidden_dims[i]))
            layers.append(nn.ReLU())
            current_dim = hidden_dims[i]


        layers.append(nn.Linear(current_dim, NUM_ACTIONS))

        self.layers = nn.Sequential(
            *layers
        )

    def action_logits(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x) # (batch, self.embedding.entire_embedding_dim)
        emb = self.layers(emb)

        return emb