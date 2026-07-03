import torch
from torch.nn import functional as F
from torch.optim import Adam

from .interface import IntrinsicReward, GridEncoder, build_mlp, NUM_ACTIONS


class ICM(IntrinsicReward):
    '''
    Intrinsic Curiosity Module (Pathak et al. 2017). A learned encoder ``phi``
    is shaped by an inverse model (predict the action from phi(s), phi(s')) so
    it captures agent-controllable features; a forward model predicts phi(s')
    from phi(s) and the action. Intrinsic reward is the forward prediction error
    in that feature space, which is why ICM is less distracted by
    action-independent noise than raw prediction error.
    '''

    def __init__(
        self,
        beta: float = 0.5,
        lr: float = 1e-4,
        feature_dim: int = 128,
        hidden_dims: tuple[int, ...] = (128,),
        lam: float = 0.2,
        d_obj: int = 8,
        d_color: int = 4,
        d_state: int = 2,
    ):
        self.beta = beta
        self.lam = lam
        self.encoder = GridEncoder(d_obj, d_color, d_state, hidden_dims, feature_dim)
        self.inverse = build_mlp(2 * feature_dim, hidden_dims, NUM_ACTIONS)
        self.forward_model = build_mlp(feature_dim + NUM_ACTIONS, hidden_dims, feature_dim)
        params = (
            list(self.encoder.parameters())
            + list(self.inverse.parameters())
            + list(self.forward_model.parameters())
        )
        self.opt = Adam(params, lr=lr)

    def _forward_error(self, phi: torch.Tensor, action: torch.Tensor, phi_next: torch.Tensor):
        a_oh = F.one_hot(action.long(), NUM_ACTIONS).float()
        pred_next = self.forward_model(torch.cat([phi, a_oh], dim=-1))
        return 0.5 * ((pred_next - phi_next) ** 2).mean(dim=-1)      # (N,)

    def compute_intrinsic(self, obs, action, next_obs) -> torch.Tensor:
        with torch.no_grad():
            phi = self.encoder(obs)
            phi_next = self.encoder(next_obs)
            r = self._forward_error(phi, action, phi_next)
        return r.detach()

    def update(self, obs, action, next_obs) -> dict[str, float]:
        phi = self.encoder(obs)
        phi_next = self.encoder(next_obs)

        # inverse: predict action from (phi, phi_next) -> shapes the encoder
        logits = self.inverse(torch.cat([phi, phi_next], dim=-1))
        inverse_loss = F.cross_entropy(logits, action.long())

        # forward: predict phi_next from (phi, action); target detached
        forward_loss = self._forward_error(phi, action, phi_next.detach()).mean()

        loss = (1.0 - self.lam) * inverse_loss + self.lam * forward_loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return {
            "icm_forward_loss": forward_loss.item(),
            "icm_inverse_loss": inverse_loss.item(),
        }
