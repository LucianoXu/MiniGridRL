# Intrinsic Reward Exploration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pluggable `IntrinsicReward` abstraction to MiniGridRL and wire it into PPO, with three interchangeable methods — RND, ICM, and LPM — selectable from YAML config.

**Architecture:** A small `minigridrl/intrinsic/` package defines an `IntrinsicReward` ABC with two methods (`compute_intrinsic` for per-transition bonus at collection time, `update` for training the module's own networks after the PPO update). PPO's train loop gains a seam: after `_collect_rollout` it computes the intrinsic bonus over valid transitions and folds `beta * r_int` into `buffers["rew"]` before GAE; after `_update` it calls `intrinsic.update`. This ordering (compute → PPO update → intrinsic.update) is what LPM requires. Networks reuse the existing `GridEmbedding` via a shared `GridEncoder`.

**Tech Stack:** Python, PyTorch, gymnasium (vector envs, NEXT_STEP autoreset), pytest, uv.

## Global Constraints

- Observation space is fixed: MiniGrid 7×7×3 `uint8` grid; action space `Discrete(7)` (`NUM_ACTIONS = 7`). Copied from `SPEC.md`.
- `GridEmbedding` consumes **integer** grid indices (it calls `x.long()` and does `nn.Embedding` lookups) — never standardize raw obs before embedding.
- Must run on Mac and Linux; detect/use MPS or CUDA when available. Current agent code is device-agnostic and runs on the default device; intrinsic modules follow the same convention (inputs and module on the same device; no forced `.cuda()`).
- Follow existing patterns: factory dispatch by string `id`, full-object pickling for checkpoints (`torch.load(..., weights_only=False)`), `tensorboard_write(**metrics)` skips `None` values.
- Run everything through the venv: prefix commands with `uv run`.
- All new tests live under `tests/` and run with `uv run pytest`.

---

## File Structure

**Create:**
- `minigridrl/intrinsic/__init__.py` — exports `IntrinsicReward`, `intrinsic_reward_factory`.
- `minigridrl/intrinsic/interface.py` — `IntrinsicReward` ABC + `RunningMeanStd` + `build_mlp` + `GridEncoder`.
- `minigridrl/intrinsic/rnd.py` — `RND`.
- `minigridrl/intrinsic/icm.py` — `ICM`.
- `minigridrl/intrinsic/lpm.py` — `LPM`.
- `minigridrl/intrinsic/factory.py` — `intrinsic_reward_factory`.
- `configs/experiments/ppo_rnd_empty5x5.yaml` — demo config.
- `tests/test_intrinsic_interface.py`, `tests/test_rnd.py`, `tests/test_ppo_intrinsic.py`, `tests/test_icm.py`, `tests/test_lpm.py`.

**Modify:**
- `minigridrl/agents/PPO.py` — add `build_next_obs` helper, `intrinsic` constructor param, `_apply_intrinsic` seam, checkpoint fields, tensorboard metrics.
- `minigridrl/agents/factory.py` — construct and inject intrinsic into PPO.

---

## Task 1: Intrinsic package skeleton — interface, RunningMeanStd, shared nets

**Files:**
- Create: `minigridrl/intrinsic/__init__.py`
- Create: `minigridrl/intrinsic/interface.py`
- Test: `tests/test_intrinsic_interface.py`

**Interfaces:**
- Consumes: `minigridrl.models.interface.GridEmbedding`, `NUM_ACTIONS`.
- Produces:
  - `class IntrinsicReward(ABC)` with attribute `beta: float`, abstract `compute_intrinsic(self, obs, action, next_obs) -> Tensor` (returns `(N,)`, detached) and `update(self, obs, action, next_obs) -> dict[str, float]`.
  - `class RunningMeanStd` with `update(x)`, properties `mean`, `var`, `std`.
  - `def build_mlp(in_dim: int, hidden_dims: Sequence[int], out_dim: int) -> nn.Sequential`.
  - `class GridEncoder(nn.Module)` with `__init__(d_obj, d_color, d_state, hidden_dims, out_dim)` and `forward(x) -> (N, out_dim)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_intrinsic_interface.py`:

```python
import torch

from minigridrl.intrinsic.interface import (
    IntrinsicReward,
    RunningMeanStd,
    build_mlp,
    GridEncoder,
)


def test_running_mean_std_matches_torch():
    rms = RunningMeanStd(shape=())
    x1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    x2 = torch.tensor([10.0, 12.0])
    rms.update(x1)
    rms.update(x2)
    allx = torch.cat([x1, x2])
    assert torch.isclose(rms.mean, allx.mean(), atol=1e-4)
    # variance is population (unbiased=False)
    assert torch.isclose(rms.var, allx.var(unbiased=False), atol=1e-3)
    assert rms.std > 0


def test_build_mlp_shapes():
    net = build_mlp(6, [8, 8], 3)
    out = net(torch.randn(5, 6))
    assert tuple(out.shape) == (5, 3)


def test_grid_encoder_forward_shape():
    enc = GridEncoder(d_obj=8, d_color=4, d_state=2, hidden_dims=[16], out_dim=12)
    obs = torch.zeros(4, 7, 7, 3, dtype=torch.uint8)
    out = enc(obs)
    assert tuple(out.shape) == (4, 12)


def test_intrinsic_reward_is_abstract():
    # cannot instantiate the ABC directly
    try:
        IntrinsicReward()
        assert False, "IntrinsicReward should be abstract"
    except TypeError:
        pass
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_intrinsic_interface.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'minigridrl.intrinsic'`.

- [ ] **Step 3: Write the interface module**

Create `minigridrl/intrinsic/interface.py`:

```python
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
```

Create `minigridrl/intrinsic/__init__.py`:

```python
from .interface import IntrinsicReward
from .factory import intrinsic_reward_factory

__all__ = ["IntrinsicReward", "intrinsic_reward_factory"]
```

> Note: `__init__.py` imports `factory`, which does not exist until Task 3's
> dependency. To keep Task 1 runnable in isolation, create a minimal stub now
> and flesh it out in the factory task.

Create `minigridrl/intrinsic/factory.py` (stub for now):

```python
from typing import Any

from .interface import IntrinsicReward


def intrinsic_reward_factory(cfg: dict[str, Any] | None) -> IntrinsicReward | None:
    '''Construct an IntrinsicReward from a config block, or None if cfg is falsy.'''
    if not cfg:
        return None
    raise ValueError("Unsupported IntrinsicReward id (no methods registered yet)")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_intrinsic_interface.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add minigridrl/intrinsic/ tests/test_intrinsic_interface.py
git commit -m "feat(intrinsic): IntrinsicReward interface, RunningMeanStd, shared nets"
```

---

## Task 2: RND

**Files:**
- Create: `minigridrl/intrinsic/rnd.py`
- Test: `tests/test_rnd.py`

**Interfaces:**
- Consumes: `IntrinsicReward`, `GridEncoder`, `RunningMeanStd` from `interface.py`.
- Produces: `class RND(IntrinsicReward)` with
  `__init__(self, beta=0.5, lr=1e-4, feature_dim=128, hidden_dims=(128,), d_obj=8, d_color=4, d_state=2)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rnd.py`:

```python
import torch

from minigridrl.intrinsic.rnd import RND


def _batch(n=16):
    torch.manual_seed(0)
    obs = torch.randint(0, 3, (n, 7, 7, 3), dtype=torch.uint8)
    action = torch.randint(0, 7, (n,))
    next_obs = torch.randint(0, 3, (n, 7, 7, 3), dtype=torch.uint8)
    return obs, action, next_obs


def test_compute_intrinsic_shape_and_detached():
    rnd = RND(beta=0.5, feature_dim=32, hidden_dims=(32,))
    obs, action, next_obs = _batch()
    r = rnd.compute_intrinsic(obs, action, next_obs)
    assert tuple(r.shape) == (16,)
    assert not r.requires_grad
    assert torch.all(r >= 0)


def test_update_reduces_error_on_fixed_batch():
    rnd = RND(beta=0.5, lr=1e-3, feature_dim=32, hidden_dims=(32,))
    obs, action, next_obs = _batch()
    # raw (unnormalized) predictor error before/after training on the same batch
    def raw_error():
        with torch.no_grad():
            t = rnd.target(next_obs)
            p = rnd.predictor(next_obs)
            return ((p - t) ** 2).mean().item()
    before = raw_error()
    for _ in range(50):
        rnd.update(obs, action, next_obs)
    after = raw_error()
    assert after < before


def test_target_is_frozen():
    rnd = RND()
    assert all(not p.requires_grad for p in rnd.target.parameters())
    assert any(p.requires_grad for p in rnd.predictor.parameters())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_rnd.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'minigridrl.intrinsic.rnd'`.

- [ ] **Step 3: Write the RND module**

Create `minigridrl/intrinsic/rnd.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_rnd.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add minigridrl/intrinsic/rnd.py tests/test_rnd.py
git commit -m "feat(intrinsic): RND (random network distillation)"
```

---

## Task 3: PPO integration seam + factory + demo config

**Files:**
- Modify: `minigridrl/agents/PPO.py`
- Modify: `minigridrl/agents/factory.py`
- Modify: `minigridrl/intrinsic/factory.py`
- Create: `configs/experiments/ppo_rnd_empty5x5.yaml`
- Test: `tests/test_ppo_intrinsic.py`

**Interfaces:**
- Consumes: `RND` (Task 2), `intrinsic_reward_factory` (this task), PPO buffers dict with keys `obs (T,N,7,7,3)`, `act (T,N)`, `rew (T,N)`, `valid (T,N) bool`.
- Produces:
  - `def build_next_obs(obs_buf: Tensor, carry_obs: Tensor) -> Tensor` in `PPO.py`.
  - `PPO.__init__(..., intrinsic: IntrinsicReward | None = None)`.
  - `PPO._apply_intrinsic(self, buffers, carry_obs) -> dict[str, float]` (mutates `buffers["rew"]`, stashes `self._int_batch`).
  - `intrinsic_reward_factory(cfg)` returning an `RND` for `id == "RND"`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ppo_intrinsic.py`:

```python
import torch

from minigridrl.agents.PPO import PPO, build_next_obs
from minigridrl.intrinsic.rnd import RND
from minigridrl.models.mlp import MLP, ValueMLP


def test_build_next_obs_shifts_and_appends_carry():
    # (T=3, N=2, 1) toy trace to check the shift/append math on a small tensor
    obs_buf = torch.arange(3 * 2 * 1).reshape(3, 2, 1).float()
    carry = torch.tensor([[100.0], [200.0]])          # (N=2, 1)
    nxt = build_next_obs(obs_buf, carry)
    assert tuple(nxt.shape) == (3, 2, 1)
    assert torch.equal(nxt[0], obs_buf[1])            # next of step 0 is step 1's obs
    assert torch.equal(nxt[1], obs_buf[2])
    assert torch.equal(nxt[2], carry)                 # last step's next is the carry-over


def _tiny_ppo(intrinsic=None):
    policy = MLP(hidden_dims=[16])
    value_net = ValueMLP(hidden_dims=[16])
    return PPO(policy, value_net, n_epochs=1, num_minibatches=1, intrinsic=intrinsic)


def _fake_buffers(T=4, N=2):
    obs = torch.randint(0, 3, (T, N, 7, 7, 3), dtype=torch.uint8)
    return {
        "obs": obs,
        "act": torch.randint(0, 7, (T, N)),
        "rew": torch.zeros(T, N),
        "valid": torch.ones(T, N, dtype=torch.bool),
    }


def test_apply_intrinsic_adds_bonus_to_valid_rewards():
    agent = _tiny_ppo(intrinsic=RND(beta=1.0, feature_dim=16, hidden_dims=(16,)))
    buffers = _fake_buffers()
    carry = torch.randint(0, 3, (2, 7, 7, 3), dtype=torch.uint8)
    metrics = agent._apply_intrinsic(buffers, carry)
    # rewards started at zero; intrinsic bonus is non-negative and generally > 0
    assert torch.all(buffers["rew"] >= 0)
    assert buffers["rew"].sum() > 0
    assert "intrinsic_reward_mean" in metrics and "beta" in metrics
    # the valid transition batch was stashed for the post-update training call
    assert agent._int_batch is not None


def test_apply_intrinsic_masks_invalid_steps():
    agent = _tiny_ppo(intrinsic=RND(beta=1.0, feature_dim=16, hidden_dims=(16,)))
    buffers = _fake_buffers()
    buffers["valid"][0, 0] = False                    # mark one step invalid
    carry = torch.randint(0, 3, (2, 7, 7, 3), dtype=torch.uint8)
    agent._apply_intrinsic(buffers, carry)
    assert buffers["rew"][0, 0].item() == 0.0         # masked step keeps zero reward
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ppo_intrinsic.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_next_obs'` (and `PPO.__init__` has no `intrinsic`).

- [ ] **Step 3: Add `build_next_obs` and the intrinsic seam to `PPO.py`**

In `minigridrl/agents/PPO.py`, add this helper just below `compute_gae` (after line 63):

```python
def build_next_obs(obs_buf: torch.Tensor, carry_obs: torch.Tensor) -> torch.Tensor:
    '''
    Build the per-step next-observation buffer from the stored obs trace.

    ``obs_buf`` is ``(T, N, ...)`` (obs at each step); ``carry_obs`` is the
    ``(N, ...)`` observation returned after the final step. Under NEXT_STEP
    autoreset ``obs_buf[t+1]`` is the correct next observation for step ``t``
    (the true terminal observation when step ``t`` is done); the trailing dummy
    step is masked out of losses via ``valid``.
    '''
    return torch.cat([obs_buf[1:], carry_obs[None]], dim=0)
```

Add the `intrinsic` parameter to `PPO.__init__`. Change the signature (line 84-106) to include the new keyword after `target_kl`:

```python
        target_kl: float | None = None,
        intrinsic=None,
    ):
```

And store it at the end of `__init__` (after `self.opt_value = ...`, around line 137):

```python
        self.intrinsic = intrinsic
        self._int_batch = None
```

Add the seam method just above `def train` (before line 380):

```python
    def _apply_intrinsic(self, buffers, carry_obs) -> dict[str, float]:
        '''
        Compute the intrinsic bonus over valid transitions and fold
        ``beta * r_int`` into ``buffers["rew"]`` in place, before GAE. The valid
        transition batch is stashed on ``self._int_batch`` so the post-update
        ``intrinsic.update`` call (in ``train``) sees the same data — preserving
        the compute-before-update ordering LPM requires.
        '''
        next_obs = build_next_obs(buffers["obs"], carry_obs)     # (T, N, ...)
        valid = buffers["valid"].reshape(-1)                     # (T*N,)
        obs_v = buffers["obs"].reshape(-1, *buffers["obs"].shape[2:])[valid]
        act_v = buffers["act"].reshape(-1)[valid]
        next_v = next_obs.reshape(-1, *next_obs.shape[2:])[valid]

        r_int = self.intrinsic.compute_intrinsic(obs_v, act_v, next_v)  # (M,)

        flat = buffers["rew"].reshape(-1)                        # view; in-place edit
        flat[valid] = flat[valid] + self.intrinsic.beta * r_int

        self._int_batch = (obs_v, act_v, next_v)
        return {
            "intrinsic_reward_mean": r_int.mean().item(),
            "intrinsic_reward_std": r_int.std().item() if r_int.numel() > 1 else 0.0,
            "beta": float(self.intrinsic.beta),
        }
```

- [ ] **Step 4: Run the seam tests to verify they pass**

Run: `uv run pytest tests/test_ppo_intrinsic.py -v`
Expected: 4 passed.

- [ ] **Step 5: Wire the seam into `train`, checkpoints, and tensorboard**

In `PPO.train`, replace the block that currently reads (lines 438-448):

```python
            buffers, obs, prev_done, stats = self._collect_rollout(
                envs, obs, prev_done, n_steps, n_envs
            )

            collected = n_steps * n_envs
            new_timesteps += collected
            self.timesteps += collected
            progress_bar.update(collected)

            diag = self._update(buffers)
            n_updates += 1
```

with:

```python
            buffers, obs, prev_done, stats = self._collect_rollout(
                envs, obs, prev_done, n_steps, n_envs
            )

            collected = n_steps * n_envs
            new_timesteps += collected
            self.timesteps += collected
            progress_bar.update(collected)

            # intrinsic bonus folded into rewards BEFORE GAE (no-op if disabled)
            int_metrics: dict[str, float] = {}
            if self.intrinsic is not None:
                int_metrics = self._apply_intrinsic(buffers, obs)

            diag = self._update(buffers)

            # train the intrinsic module AFTER the PPO update (LPM ordering)
            if self.intrinsic is not None:
                obs_v, act_v, next_v = self._int_batch
                int_metrics.update(self.intrinsic.update(obs_v, act_v, next_v))

            n_updates += 1
```

Then pass the intrinsic metrics into the tensorboard call. Change the end of the
`self.tensorboard_write(...)` call (line 493, the `n_updates=n_updates,` line) so
the closing looks like:

```python
                n_updates=n_updates,
                **int_metrics,
            )
```

Add the intrinsic object to the checkpoint. In `save_local`, add one entry to the
saved dict (after `"timesteps": self.timesteps,`, line 173):

```python
                "intrinsic": self.intrinsic,
```

In `load_local`, pass it back into the constructor. Change the `cls(...)` call
(lines 190-194) to:

```python
        agent = cls(
            policy=ckpt["policy"],
            value_net=ckpt["value_net"],
            intrinsic=ckpt.get("intrinsic"),
            **ckpt["hyperparams"],
        )
```

- [ ] **Step 6: Implement the real `intrinsic_reward_factory`**

Replace the body of `minigridrl/intrinsic/factory.py`:

```python
from typing import Any

from .interface import IntrinsicReward


def intrinsic_reward_factory(cfg: dict[str, Any] | None) -> IntrinsicReward | None:
    '''
    Construct an IntrinsicReward from an ``agent.intrinsic_reward`` config block.
    Returns None when cfg is falsy (missing/empty) -> pure extrinsic training.
    '''
    if not cfg:
        return None

    if "id" not in cfg:
        raise ValueError("intrinsic_reward config must contain an 'id' field")

    id = cfg["id"]
    common = {k: v for k, v in cfg.items() if k != "id"}

    if id == "RND":
        from .rnd import RND
        return RND(**common)

    raise ValueError("Unsupported IntrinsicReward ID: " + str(id))
```

- [ ] **Step 7: Inject intrinsic in the agent factory**

In `minigridrl/agents/factory.py`, inside the `elif id == 'PPO':` branch, build the
intrinsic module and pass it to `PPO(...)`. Add before the `agent = PPO(` line:

```python
        from ..intrinsic import intrinsic_reward_factory
        intrinsic = intrinsic_reward_factory(cfg.get('intrinsic_reward'))
```

and add one argument to the `PPO(...)` constructor call (after `target_kl=...`):

```python
            target_kl = cfg.get('target_kl', None),
            intrinsic = intrinsic,
        )
```

- [ ] **Step 8: Create the demo config**

Create `configs/experiments/ppo_rnd_empty5x5.yaml`:

```yaml
config_type: rl_train

working_dir: trials/ppo_rnd_empty5x5

rl_training_dir: rl_training

# ---- environment (env_factory): describes ONE env; n_envs lives in training_cfg ----
envs:
  id: MiniGrid-Empty-5x5-v0
  disable_env_checker: True

# ---- agent (agent_factory -> PPO) ----
agent:
  id: PPO
  lr: 0.0003
  betas: [0.9, 0.999]
  eps: 1.0e-8
  weight_decay: 0.0
  grad_norm_clip: 0.5

  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  clip_range_vf: null
  n_epochs: 10
  num_minibatches: 4
  vf_coef: 0.5
  ent_coef: 0.01
  normalize_advantage: true
  target_kl: 0.02

  model:
    id: MLP
    d_obj: 8
    d_color: 4
    d_state: 2
    hidden_dims: [96, 48]

  value_model:
    id: ValueMLP
    d_obj: 8
    d_color: 4
    d_state: 2
    hidden_dims: [96, 48]

  # ---- intrinsic reward (intrinsic_reward_factory -> RND); omit for pure extrinsic ----
  intrinsic_reward:
    id: RND
    beta: 0.5
    lr: 0.0001
    feature_dim: 128
    hidden_dims: [128]
    d_obj: 8
    d_color: 4
    d_state: 2

# ---- passed as **kwargs to PPO.train() ----
training_cfg:
  num_timesteps: 100000
  n_envs: 16
  n_steps: 128
  save_interval: null
  video_interval: 20000
  video_fps: 8
  seed: 42
```

- [ ] **Step 9: Run the full test suite and a short smoke train**

Run: `uv run pytest tests/ -v`
Expected: all pass (existing + new).

Run a short smoke test to confirm the config trains end-to-end and RND is active:

```bash
uv run python -c "from minigridrl import run; run({'config_type':'rl_train','working_dir':'/tmp/ppo_rnd_smoke','rl_training_dir':'rl','envs':{'id':'MiniGrid-Empty-5x5-v0','disable_env_checker':True},'agent':{'id':'PPO','lr':3e-4,'betas':[0.9,0.999],'eps':1e-8,'weight_decay':0.0,'grad_norm_clip':0.5,'gamma':0.99,'gae_lambda':0.95,'clip_range':0.2,'clip_range_vf':None,'n_epochs':2,'num_minibatches':2,'vf_coef':0.5,'ent_coef':0.01,'normalize_advantage':True,'target_kl':None,'model':{'id':'MLP','d_obj':8,'d_color':4,'d_state':2,'hidden_dims':[32]},'value_model':{'id':'ValueMLP','d_obj':8,'d_color':4,'d_state':2,'hidden_dims':[32]},'intrinsic_reward':{'id':'RND','beta':0.5,'lr':1e-4,'feature_dim':32,'hidden_dims':[32]}},'training_cfg':{'num_timesteps':2000,'n_envs':4,'n_steps':64,'seed':0}}, overwrite=True)"
```

Expected: training runs to completion, prints progress, no exceptions. Clean up: `rm -rf /tmp/ppo_rnd_smoke`.

- [ ] **Step 10: Commit**

```bash
git add minigridrl/agents/PPO.py minigridrl/agents/factory.py minigridrl/intrinsic/factory.py configs/experiments/ppo_rnd_empty5x5.yaml tests/test_ppo_intrinsic.py
git commit -m "feat(ppo): intrinsic-reward seam + RND factory wiring + demo config"
```

---

## Task 4: ICM

**Files:**
- Create: `minigridrl/intrinsic/icm.py`
- Modify: `minigridrl/intrinsic/factory.py`
- Test: `tests/test_icm.py`

**Interfaces:**
- Consumes: `IntrinsicReward`, `GridEncoder`, `build_mlp`, `NUM_ACTIONS` from `interface.py`.
- Produces: `class ICM(IntrinsicReward)` with
  `__init__(self, beta=0.5, lr=1e-4, feature_dim=128, hidden_dims=(128,), lam=0.2, d_obj=8, d_color=4, d_state=2)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_icm.py`:

```python
import torch

from minigridrl.intrinsic.icm import ICM


def _batch(n=16):
    torch.manual_seed(1)
    obs = torch.randint(0, 3, (n, 7, 7, 3), dtype=torch.uint8)
    action = torch.randint(0, 7, (n,))
    next_obs = torch.randint(0, 3, (n, 7, 7, 3), dtype=torch.uint8)
    return obs, action, next_obs


def test_compute_intrinsic_shape_and_detached():
    icm = ICM(beta=0.5, feature_dim=32, hidden_dims=(32,))
    obs, action, next_obs = _batch()
    r = icm.compute_intrinsic(obs, action, next_obs)
    assert tuple(r.shape) == (16,)
    assert not r.requires_grad
    assert torch.all(r >= 0)


def test_update_returns_forward_and_inverse_losses():
    icm = ICM(beta=0.5, lr=1e-3, feature_dim=32, hidden_dims=(32,))
    obs, action, next_obs = _batch()
    metrics = icm.update(obs, action, next_obs)
    assert "icm_forward_loss" in metrics
    assert "icm_inverse_loss" in metrics


def test_update_reduces_forward_loss_on_fixed_batch():
    icm = ICM(beta=0.5, lr=1e-3, feature_dim=32, hidden_dims=(32,))
    obs, action, next_obs = _batch()
    before = icm.update(obs, action, next_obs)["icm_forward_loss"]
    for _ in range(50):
        after = icm.update(obs, action, next_obs)["icm_forward_loss"]
    assert after < before
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_icm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'minigridrl.intrinsic.icm'`.

- [ ] **Step 3: Write the ICM module**

Create `minigridrl/intrinsic/icm.py`:

```python
import torch
from torch import nn
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
```

- [ ] **Step 4: Register ICM in the factory**

In `minigridrl/intrinsic/factory.py`, add before the final `raise`:

```python
    if id == "ICM":
        from .icm import ICM
        return ICM(**common)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_icm.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add minigridrl/intrinsic/icm.py minigridrl/intrinsic/factory.py tests/test_icm.py
git commit -m "feat(intrinsic): ICM (intrinsic curiosity module)"
```

---

## Task 5: LPM

**Files:**
- Create: `minigridrl/intrinsic/lpm.py`
- Modify: `minigridrl/intrinsic/factory.py`
- Test: `tests/test_lpm.py`

**Interfaces:**
- Consumes: `IntrinsicReward`, `GridEncoder`, `build_mlp`, `NUM_ACTIONS` from `interface.py`.
- Produces: `class LPM(IntrinsicReward)` with
  `__init__(self, beta=0.5, lr=1e-3, feature_dim=128, hidden_dims=(128,), queue_size=100, buffer_size=2048, batch_size=256, d_obj=8, d_color=4, d_state=2)`.

**Design recap (from spec):** dynamics `f` predicts the *next feature* `psi(next_obs)` from
`[psi(obs), onehot(a)]`, where `psi` is a **fixed random** `GridEncoder`. Per-step error is
`eps = log MSE(psi(next_obs), f(...))`. Intrinsic reward is `g(psi(obs), a) - eps`, where the
error model `g` predicts the *previous* dynamics model's expected error (fit against the
fixed-size queue `D` of stored `eps`). Reward is 0 until `D` is full (warmup). `compute_intrinsic`
uses the current (pre-update) dynamics and pushes into `B`/`D`; `update` then trains `f` and `g`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_lpm.py`:

```python
import torch

from minigridrl.intrinsic.lpm import LPM


def _batch(n=32, seed=2):
    torch.manual_seed(seed)
    obs = torch.randint(0, 3, (n, 7, 7, 3), dtype=torch.uint8)
    action = torch.randint(0, 7, (n,))
    next_obs = torch.randint(0, 3, (n, 7, 7, 3), dtype=torch.uint8)
    return obs, action, next_obs


def test_reward_is_zero_during_warmup():
    # queue_size larger than one batch -> first call is pure warmup -> all zeros
    lpm = LPM(feature_dim=32, hidden_dims=(32,), queue_size=64)
    obs, action, next_obs = _batch(n=32)
    r = lpm.compute_intrinsic(obs, action, next_obs)
    assert tuple(r.shape) == (32,)
    assert torch.all(r == 0)


def test_reward_nonzero_after_queue_fills():
    lpm = LPM(feature_dim=32, hidden_dims=(32,), queue_size=32)
    obs, action, next_obs = _batch(n=32)
    lpm.compute_intrinsic(obs, action, next_obs)   # fills D to 32
    lpm.update(obs, action, next_obs)
    r = lpm.compute_intrinsic(obs, action, next_obs)   # D now full
    assert not r.requires_grad
    assert torch.any(r != 0)


def test_update_reduces_dynamics_loss_on_fixed_batch():
    lpm = LPM(feature_dim=32, hidden_dims=(32,), queue_size=32, batch_size=32)
    obs, action, next_obs = _batch(n=32)
    lpm.compute_intrinsic(obs, action, next_obs)
    before = lpm.update(obs, action, next_obs)["lpm_dynamics_loss"]
    for _ in range(50):
        lpm.compute_intrinsic(obs, action, next_obs)
        after = lpm.update(obs, action, next_obs)["lpm_dynamics_loss"]
    assert after < before


def test_error_model_is_fixed_encoder_frozen():
    lpm = LPM(feature_dim=16, hidden_dims=(16,))
    assert all(not p.requires_grad for p in lpm.psi.parameters())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_lpm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'minigridrl.intrinsic.lpm'`.

- [ ] **Step 3: Write the LPM module**

Create `minigridrl/intrinsic/lpm.py`:

```python
import torch
from torch import nn
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
```

- [ ] **Step 4: Register LPM in the factory**

In `minigridrl/intrinsic/factory.py`, add before the final `raise`:

```python
    if id == "LPM":
        from .lpm import LPM
        return LPM(**common)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_lpm.py -v`
Expected: 4 passed.

- [ ] **Step 6: Full suite + end-to-end smoke for all three methods**

Run: `uv run pytest tests/ -v`
Expected: all pass.

Confirm each method trains end-to-end through the factory (swap `id`):

```bash
uv run python -c "
from minigridrl import run
base = {'config_type':'rl_train','rl_training_dir':'rl','envs':{'id':'MiniGrid-Empty-5x5-v0','disable_env_checker':True},'agent':{'id':'PPO','lr':3e-4,'betas':[0.9,0.999],'eps':1e-8,'weight_decay':0.0,'grad_norm_clip':0.5,'gamma':0.99,'gae_lambda':0.95,'clip_range':0.2,'clip_range_vf':None,'n_epochs':2,'num_minibatches':2,'vf_coef':0.5,'ent_coef':0.01,'normalize_advantage':True,'target_kl':None,'model':{'id':'MLP','d_obj':8,'d_color':4,'d_state':2,'hidden_dims':[32]},'value_model':{'id':'ValueMLP','d_obj':8,'d_color':4,'d_state':2,'hidden_dims':[32]}},'training_cfg':{'num_timesteps':2000,'n_envs':4,'n_steps':64,'seed':0}}
for m in [{'id':'ICM','beta':0.5,'lr':1e-4,'feature_dim':32,'hidden_dims':[32]},{'id':'LPM','beta':0.5,'lr':1e-3,'feature_dim':32,'hidden_dims':[32],'queue_size':64,'batch_size':64}]:
    cfg = {**base, 'working_dir': f'/tmp/ppo_{m[\"id\"]}_smoke', 'agent': {**base['agent'], 'intrinsic_reward': m}}
    run(cfg, overwrite=True)
    print('OK', m['id'])
"
```

Expected: prints `OK ICM` and `OK LPM`, no exceptions. Clean up: `rm -rf /tmp/ppo_ICM_smoke /tmp/ppo_LPM_smoke`.

- [ ] **Step 7: Commit**

```bash
git add minigridrl/intrinsic/lpm.py minigridrl/intrinsic/factory.py tests/test_lpm.py
git commit -m "feat(intrinsic): LPM (learning progress monitoring)"
```

---

## Self-Review Notes

- **Spec coverage:** interface (Task 1) ✓; RunningMeanStd/GridEncoder shared tools (Task 1) ✓; RND (Task 2) ✓; PPO seam with next_obs construction + compute→update ordering + single combined return + checkpoint pickling + tensorboard metrics + factory/config (Task 3) ✓; ICM (Task 4) ✓; LPM dual buffer + fixed encoder + warmup + previous-model error fit (Task 5) ✓. Open spec items intentionally out of scope: dual value head, noisy-TV wrapper, episodic-bonus family.
- **Type consistency:** `compute_intrinsic(obs, action, next_obs)` and `update(obs, action, next_obs)` signatures identical across RND/ICM/LPM and the PPO seam; `intrinsic_reward_factory(cfg) -> IntrinsicReward | None`; `build_next_obs(obs_buf, carry_obs)` used only in `_apply_intrinsic`; metric keys (`rnd_loss`, `icm_forward_loss`, `icm_inverse_loss`, `lpm_dynamics_loss`, `lpm_error_loss`, `intrinsic_reward_mean/std`, `beta`) match the spec's tensorboard list.
- **LPM update frequency:** update cycle N is aligned to one-per-rollout (open question #1 default). Change requires a sub-rollout counter in the seam; deferred unless requested.
