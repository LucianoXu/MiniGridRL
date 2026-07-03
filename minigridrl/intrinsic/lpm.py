import torch
from torch.nn import functional as F
from torch.optim import Adam

from .interface import IntrinsicReward, GridEncoder, build_mlp, NUM_ACTIONS


class _RingBuffer:
    '''Fixed-capacity tensor store; appends batches, trims oldest, samples rows.'''

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._cols: list[torch.Tensor] | None = None   # one tensor per column

    def __len__(self) -> int:
        return 0 if self._cols is None else self._cols[0].shape[0]

    def push(self, *columns: torch.Tensor) -> None:
        cols = [c.detach() for c in columns]
        if self._cols is None:
            self._cols = cols
        else:
            self._cols = [torch.cat([old, new], dim=0) for old, new in zip(self._cols, cols)]
        if len(self) > self.capacity:
            self._cols = [c[-self.capacity:] for c in self._cols]

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        n = len(self)
        idx = torch.randint(0, n, (min(batch_size, n),))
        return tuple(c[idx] for c in self._cols)


class LPM(IntrinsicReward):
    '''
    Learning Progress Monitoring (Hou et al. ICLR 2026). Rewards model
    improvement rather than prediction error, which is naturally robust to
    noisy-TV (unlearnable) transitions.

    MiniGrid adaptation: a fixed random encoder ``psi`` maps observations to a
    feature space; the dynamics model ``f`` predicts ``psi(next_obs)`` and the
    error model ``g`` predicts the *previous* dynamics model's expected log-MSE.
    Intrinsic reward is ``g(psi(s), a) - eps`` where ``eps`` is the current
    model's log-MSE. See the design spec for the theory (Thm 4.1/4.2).
    '''

    def __init__(
        self,
        beta: float = 0.5,
        lr: float = 1e-3,
        feature_dim: int = 128,
        hidden_dims: tuple[int, ...] = (128,),
        queue_size: int = 100,
        buffer_size: int = 2048,
        batch_size: int = 256,
        d_obj: int = 8,
        d_color: int = 4,
        d_state: int = 2,
    ):
        self.beta = beta
        self.queue_size = queue_size
        self.batch_size = batch_size

        # fixed random encoder -> stable prediction target, no collapse
        self.psi = GridEncoder(d_obj, d_color, d_state, hidden_dims, feature_dim)
        for p in self.psi.parameters():
            p.requires_grad_(False)

        self.dynamics = build_mlp(feature_dim + NUM_ACTIONS, hidden_dims, feature_dim)
        self.error_model = build_mlp(feature_dim + NUM_ACTIONS, hidden_dims, 1)
        self.opt_dyn = Adam(self.dynamics.parameters(), lr=lr)
        self.opt_err = Adam(self.error_model.parameters(), lr=lr)

        self.B = _RingBuffer(buffer_size)   # (obs, act, next_obs) for dynamics
        self.D = _RingBuffer(queue_size)    # (obs, act, eps) for error model

    def _feat_action(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        z = self.psi(obs)
        a_oh = F.one_hot(action.long(), NUM_ACTIONS).float()
        return torch.cat([z, a_oh], dim=-1)

    def _log_mse(self, obs, action, next_obs) -> torch.Tensor:
        '''Per-sample log MSE between predicted and true next feature, shape (N,).'''
        pred = self.dynamics(self._feat_action(obs, action))
        target = self.psi(next_obs)
        mse = ((pred - target) ** 2).mean(dim=-1)
        return torch.log(mse + 1e-8)

    def compute_intrinsic(self, obs, action, next_obs) -> torch.Tensor:
        with torch.no_grad():
            eps = self._log_mse(obs, action, next_obs)              # (N,) current model
            g_pred = self.error_model(self._feat_action(obs, action)).reshape(-1)

        if len(self.D) >= self.queue_size:
            r = g_pred - eps                                        # learning progress
        else:
            r = torch.zeros_like(eps)                              # warmup

        self.B.push(obs, action, next_obs)
        self.D.push(obs, action, eps)
        return r.detach()

    def update(self, obs, action, next_obs) -> dict[str, float]:
        # dynamics model on a sampled batch from B
        b_obs, b_act, b_next = self.B.sample(self.batch_size)
        dyn_loss = self._log_mse(b_obs, b_act, b_next).exp().mean()  # MSE = exp(log MSE)
        self.opt_dyn.zero_grad()
        dyn_loss.backward()
        self.opt_dyn.step()

        # error model fits the stored eps (computed with the PREVIOUS dynamics)
        d_obs, d_act, d_eps = self.D.sample(self.batch_size)
        g_pred = self.error_model(self._feat_action(d_obs, d_act)).reshape(-1)
        err_loss = F.mse_loss(g_pred, d_eps)
        self.opt_err.zero_grad()
        err_loss.backward()
        self.opt_err.step()

        return {
            "lpm_dynamics_loss": dyn_loss.item(),
            "lpm_error_loss": err_loss.item(),
        }
