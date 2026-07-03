from abc import ABC, abstractmethod
from typing import Sequence

import torch
from torch import nn

from ..models.interface import GridEmbedding, NUM_ACTIONS  # noqa: F401  (re-export convenience)


class RunningMeanStd:
    '''
    Streaming mean/variance via Chan's parallel algorithm (OpenAI-baselines
    style). Tracks statistics over the leading (batch) dimension of the inputs.
    '''

    def __init__(self, shape: tuple = (), epsilon: float = 1e-4):
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = float(epsilon)

    def update(self, x: torch.Tensor) -> None:
        x = x.detach()
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    @property
    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var + 1e-8)


def build_mlp(in_dim: int, hidden_dims: Sequence[int], out_dim: int) -> nn.Sequential:
    '''Plain ReLU MLP: in_dim -> hidden... -> out_dim (no activation on output).'''
    layers: list[nn.Module] = []
    current = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(current, h))
        layers.append(nn.ReLU())
        current = h
    layers.append(nn.Linear(current, out_dim))
    return nn.Sequential(*layers)


class GridEncoder(nn.Module):
    '''
    Reusable observation encoder for intrinsic modules: the shared
    ``GridEmbedding`` (integer grid -> flat embedding) followed by an MLP to a
    fixed-size feature vector. Used as RND target/predictor, ICM encoder, and
    LPM's fixed random encoder.
    '''

    def __init__(
        self,
        d_obj: int = 8,
        d_color: int = 4,
        d_state: int = 2,
        hidden_dims: Sequence[int] = (128,),
        out_dim: int = 128,
    ):
        super().__init__()
        self.embedding = GridEmbedding(d_obj, d_color, d_state)
        self.mlp = build_mlp(self.embedding.entire_embedding_dim, hidden_dims, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.embedding(x))


class IntrinsicReward(ABC):
    '''
    Pluggable intrinsic-reward module. Two-method contract, ordering-sensitive:
    the agent calls ``compute_intrinsic`` at collection time (using the current
    networks), runs its own RL update, then calls ``update`` to train these
    networks. LPM depends on that order.

    Checkpointing: the whole object is pickled by the agent's ``save_local``
    (same as ``policy``/``value_net``), so no explicit ``state_dict`` is needed.
    '''

    beta: float

    @abstractmethod
    def compute_intrinsic(
        self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor
    ) -> torch.Tensor:
        '''Per-transition intrinsic reward, shape (N,), detached. No policy grad.'''
        ...

    @abstractmethod
    def update(
        self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor
    ) -> dict[str, float]:
        '''Train own networks on the collected transitions; return scalar metrics.'''
        ...
