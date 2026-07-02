from abc import ABC, abstractmethod
import torch
from torch import nn

# cardinals of three channels of MiniGrid Image Input
NUM_OBJECTS, NUM_COLORS, NUM_STATES = 11, 6, 3
NUM_ACTIONS = 7

class GridEmbedding(nn.Module):
    def __init__(self, d_obj: int = 8, d_color: int = 4, d_state: int = 2):
        super().__init__()
        self.obj_emb   = nn.Embedding(NUM_OBJECTS, d_obj)
        self.color_emb = nn.Embedding(NUM_COLORS,  d_color)
        self.state_emb = nn.Embedding(NUM_STATES,  d_state)
        self.embed_dim = d_obj + d_color + d_state  # the embedding dimension of each cell
        self.entire_embedding_dim = 7 * 7 * self.embed_dim  # the embdding dimension of the flattened entire observation

    def forward(self, x: torch.ByteTensor) -> torch.Tensor:
        # x: (batch, 7, 7, 3)
        _x = x.long()
        obj   = self.obj_emb(_x[..., 0])               # (batch, 7, 7, d_obj)
        color = self.color_emb(_x[..., 1])               # (batch, 7, 7, d_color)
        state = self.state_emb(_x[..., 2])               # (batch, 7, 7, d_state)
        concat = torch.cat([obj, color, state], dim=-1)   # (batch, 7, 7, embed_dim)
        return concat.reshape(-1, self.entire_embedding_dim)


class RLModule(ABC, nn.Module):
    
    @abstractmethod
    def action_logits(self, x: torch.Tensor) -> torch.Tensor:
        '''
        output the logits for different actions
        '''
        ...

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        input:
            x: batched observation
            temperature: softmax temperature. >1 flattens the distribution
                (more exploration), <1 sharpens it (more greedy). Must be > 0.

        output: tuple of batched
            (sampled actions, log_probabilities of those actions, distribution entropy)
        '''

        logits = self.action_logits(x) / temperature

        dist = torch.distributions.Categorical(logits=logits)

        actions = dist.sample()              # (batch,) sampled from the (tempered) policy
        log_prob = dist.log_prob(actions)    # (batch,) log pi(a|s) of the taken action
        entropy = dist.entropy()             # (batch,) H(pi(.|s)), exploration diagnostic

        return actions, log_prob, entropy