import torch
from torch.optim import Adam

from .interface import IntrinsicReward, GridEncoder, RunningMeanStd


class RND(IntrinsicReward):
    '''
    Random Network Distillation (Burda et al. 2018). A fixed random ``target``
    encoder and a trained ``predictor`` map next-observations to feature space;
    the intrinsic reward is the squared predictor error, novelty as surprise.

    MiniGrid note: raw-obs standardization from the original paper is dropped
    (GridEmbedding needs integer indices); only the intrinsic reward is
    normalized by its running std so ``beta`` stays comparable across envs.
    '''

    def __init__(
        self,
        beta: float = 0.5,
        lr: float = 1e-4,
        feature_dim: int = 128,
        hidden_dims: tuple[int, ...] = (128,),
        d_obj: int = 8,
        d_color: int = 4,
        d_state: int = 2,
    ):
        self.beta = beta
        self.target = GridEncoder(d_obj, d_color, d_state, hidden_dims, feature_dim)
        self.predictor = GridEncoder(d_obj, d_color, d_state, hidden_dims, feature_dim)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.opt = Adam(self.predictor.parameters(), lr=lr)
        self.rew_rms = RunningMeanStd(shape=())

    def _error(self, next_obs: torch.Tensor) -> torch.Tensor:
        '''Per-sample squared predictor error over the feature dim, shape (N,).'''
        target = self.target(next_obs)
        pred = self.predictor(next_obs)
        return ((pred - target) ** 2).mean(dim=-1)

    def compute_intrinsic(self, obs, action, next_obs) -> torch.Tensor:
        with torch.no_grad():
            err = self._error(next_obs)          # (N,)
        self.rew_rms.update(err)
        return (err / self.rew_rms.std).detach()

    def update(self, obs, action, next_obs) -> dict[str, float]:
        loss = self._error(next_obs).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return {"rnd_loss": loss.item()}
