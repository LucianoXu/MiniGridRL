from pathlib import Path
from typing import Callable
import time

import gymnasium as gym
from gymnasium.vector import AutoresetMode

import torch
from torch import nn
from torch.optim import Adam
from torch.distributions import Categorical

from .interface import RLAgent
from ..models.interface import RLModule
from ..render import record_rollout_video

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    bootstrap_value: torch.Tensor,
    terminateds: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    '''
    Generalized Advantage Estimation over a fixed-horizon rollout.

    Shapes: ``rewards``, ``values``, ``terminateds``, ``dones`` are ``(T, N)``
    (time-major, N parallel envs); ``bootstrap_value`` is ``(N,)`` -- the value
    of the observation that follows the last rollout step.

    Two boundary flags are kept SEPARATE, which is what makes truncation correct
    under ``NEXT_STEP`` autoreset:

    * ``terminateds[t]``: 1.0 only on genuine termination (goal reached). The
      terminal state's value is 0, so the bootstrap term is zeroed.
    * ``dones[t]``: terminated OR truncated. Any episode boundary resets the GAE
      trace so advantages never leak across episodes.

    For a truncated step (``done`` but not ``terminated``) the bootstrap is kept:
    ``values[t+1]`` is the value of the true terminal observation, because
    ``NEXT_STEP`` autoreset returns that observation as the next step's input.

        delta[t] = r[t] + gamma * V(s[t+1]) * (1 - terminated[t]) - V(s[t])
        A[t]     = delta[t] + gamma * lambda * (1 - done[t]) * A[t+1]
    '''
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(N, dtype=rewards.dtype)

    for t in reversed(range(T)):
        next_value = bootstrap_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * (1.0 - terminateds[t]) - values[t]
        last_gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * last_gae
        advantages[t] = last_gae

    return advantages


class PPO(RLAgent):
    '''
    Proximal Policy Optimization (clipped) with a separate policy and value
    network, GAE, and fixed-horizon rollout collection under NEXT_STEP autoreset.

    Design notes
    ------------
    * ``policy`` is the same actor ``RLModule`` REINFORCE uses. Its ``forward``
      samples actions during collection; its ``action_logits`` is re-used during
      the update epochs to evaluate log-probs / entropy of the stored actions.
    * ``value_net`` is an independent critic ``obs -> V(obs)``; keeping it
      separate avoids policy/value gradient interference (SB3's default too).
    * The step immediately after a ``done`` is a NEXT_STEP autoreset "dummy"
      step: the action is ignored and reward is 0. Those steps are collected
      (their value is the terminal-observation value used to bootstrap truncated
      episodes) but masked out of every loss via ``valid``.
    '''

    def __init__(
        self,
        policy: RLModule,
        value_net: nn.Module,

        lr = 0.0003,
        betas = (0.9, 0.999),
        eps = 1e-8,
        weight_decay = 0.,
        grad_norm_clip = 0.5,

        # ---- PPO algorithm hyperparameters ----
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: float | None = None,
        n_epochs: int = 10,
        num_minibatches: int = 4,
        vf_coef: float = 0.5,
        ent_coef: float = 0.0,
        normalize_advantage: bool = True,
        target_kl: float | None = None,
    ):
        super().__init__()

        self.policy = policy
        self.value_net = value_net
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.grad_norm_clip = grad_norm_clip

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.n_epochs = n_epochs
        self.num_minibatches = num_minibatches
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        self.timesteps = 0

        self.opt_policy = Adam(
            policy.parameters(),
            lr = lr, betas = betas, eps = eps, weight_decay = weight_decay,
        )
        self.opt_value = Adam(
            value_net.parameters(),
            lr = lr, betas = betas, eps = eps, weight_decay = weight_decay,
        )

    # ---- hyperparameters that round-trip through a checkpoint ----
    def _hyperparams(self) -> dict:
        return {
            "lr": self.lr,
            "betas": self.betas,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "grad_norm_clip": self.grad_norm_clip,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "clip_range_vf": self.clip_range_vf,
            "n_epochs": self.n_epochs,
            "num_minibatches": self.num_minibatches,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "normalize_advantage": self.normalize_advantage,
            "target_kl": self.target_kl,
        }

    def save_local(self, pt_path: str | Path):
        '''
        Save the policy and value modules (architecture + weights), both
        optimizer states, and the hyperparameters into pt_path.
        '''
        pt_path = Path(pt_path)
        pt_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "policy": self.policy,
                "value_net": self.value_net,
                "policy_optimizer_state": self.opt_policy.state_dict(),
                "value_optimizer_state": self.opt_value.state_dict(),
                "timesteps": self.timesteps,
                "hyperparams": self._hyperparams(),
            },
            pt_path,
        )

    @classmethod
    def load_local(cls, pt_path: str | Path):
        '''
        Rebuild a PPO agent from a checkpoint written by `save_local`.
        '''
        pt_path = Path(pt_path)

        # weights_only=False: the checkpoint contains pickled nn.Modules, not
        # just tensors. Only load checkpoints you trust.
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)

        agent = cls(
            policy=ckpt["policy"],
            value_net=ckpt["value_net"],
            **ckpt["hyperparams"],
        )
        agent.opt_policy.load_state_dict(ckpt["policy_optimizer_state"])
        agent.opt_value.load_state_dict(ckpt["value_optimizer_state"])
        agent.timesteps = ckpt["timesteps"]
        return agent

    def _collect_rollout(self, envs, obs, prev_done, n_steps, n_envs):
        '''
        Roll the policy forward for ``n_steps`` under NEXT_STEP autoreset,
        WITHOUT resetting between episodes. Returns the stacked buffers plus the
        carry-over ``obs`` / ``prev_done`` for the next rollout, and the list of
        completed-episode returns/lengths observed during the rollout.

        ``valid[t] = ~prev_done`` marks real transitions: the step right after a
        ``done`` is an autoreset dummy (ignored action, zero reward) and is
        masked out of all losses.
        '''
        obs_buf, act_buf, logp_buf, val_buf = [], [], [], []
        rew_buf, term_buf, done_buf, ent_buf, valid_buf = [], [], [], [], []

        # running per-env accumulators for completed-episode statistics
        running_return = torch.zeros(n_envs)
        running_len = torch.zeros(n_envs)
        ep_returns: list[float] = []
        ep_lengths: list[float] = []

        for _ in range(n_steps):
            valid = ~prev_done                                   # (N,) bool

            with torch.no_grad():
                acts, logp, ent = self.policy(obs)               # samples actions
                vals = self.value_net(obs).reshape(-1)           # (N,)

            next_obs, rewards, terminated, truncated, _ = envs.step(acts)
            rewards = rewards.float()
            terminated = terminated.bool()
            done = terminated | truncated.bool()

            obs_buf.append(obs)
            act_buf.append(acts)
            logp_buf.append(logp)
            val_buf.append(vals)
            rew_buf.append(rewards)
            term_buf.append(terminated.float())
            done_buf.append(done.float())
            ent_buf.append(ent)
            valid_buf.append(valid)

            # completed-episode bookkeeping over real (valid) steps only
            valid_f = valid.float()
            running_return += rewards * valid_f
            running_len += valid_f
            completed = done & valid
            if bool(completed.any()):
                ep_returns.extend(running_return[completed].tolist())
                ep_lengths.extend(running_len[completed].tolist())
                keep = (~completed).float()
                running_return = running_return * keep
                running_len = running_len * keep

            prev_done = done
            obs = next_obs

        with torch.no_grad():
            bootstrap_value = self.value_net(obs).reshape(-1)    # (N,)

        buffers = {
            "obs": torch.stack(obs_buf),                         # (T, N, ...)
            "act": torch.stack(act_buf),                         # (T, N)
            "logp": torch.stack(logp_buf),                       # (T, N)
            "val": torch.stack(val_buf),                         # (T, N)
            "rew": torch.stack(rew_buf),                         # (T, N)
            "term": torch.stack(term_buf),                       # (T, N)
            "done": torch.stack(done_buf),                       # (T, N)
            "ent": torch.stack(ent_buf),                         # (T, N)
            "valid": torch.stack(valid_buf),                     # (T, N) bool
            "bootstrap_value": bootstrap_value,                  # (N,)
        }
        stats = {"ep_returns": ep_returns, "ep_lengths": ep_lengths}
        return buffers, obs, prev_done, stats

    def _update(self, buffers):
        '''
        PPO clipped-surrogate update over the collected rollout: GAE advantages,
        several epochs of shuffled minibatch SGD on the valid transitions, and a
        clipped value-function regression. Returns a dict of scalar diagnostics.
        '''
        advantages = compute_gae(
            buffers["rew"], buffers["val"], buffers["bootstrap_value"],
            buffers["term"], buffers["done"], self.gamma, self.gae_lambda,
        )                                                        # (T, N)
        returns = advantages + buffers["val"]

        valid = buffers["valid"].reshape(-1)                     # (T*N,)
        obs = buffers["obs"].reshape(-1, *buffers["obs"].shape[2:])[valid]
        act = buffers["act"].reshape(-1)[valid]
        old_logp = buffers["logp"].reshape(-1)[valid]
        old_val = buffers["val"].reshape(-1)[valid]
        adv = advantages.reshape(-1)[valid]
        ret = returns.reshape(-1)[valid]

        M = obs.shape[0]
        mb_size = max(1, M // self.num_minibatches)

        pg_losses, v_losses, ent_losses, clip_fracs, approx_kls = [], [], [], [], []
        grad_norm_last = 0.0
        approx_kl = 0.0

        for _ in range(self.n_epochs):
            perm = torch.randperm(M)
            for s in range(0, M, mb_size):
                idx = perm[s:s + mb_size]
                mb_adv = adv[idx]
                if self.normalize_advantage and mb_adv.numel() > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                logits = self.policy.action_logits(obs[idx])
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(act[idx])
                entropy = dist.entropy()
                new_val = self.value_net(obs[idx]).reshape(-1)

                logratio = new_logp - old_logp[idx]
                ratio = logratio.exp()

                with torch.no_grad():
                    # http://joschu.net/blog/kl-approx.html (low-variance, >=0)
                    approx_kl = ((ratio - 1.0) - logratio).mean().item()
                    clip_fracs.append(
                        ((ratio - 1.0).abs() > self.clip_range).float().mean().item()
                    )

                # clipped policy surrogate (minimized)
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                pg_loss = torch.max(pg1, pg2).mean()

                # value regression, optionally clipped around the old value
                if self.clip_range_vf is not None:
                    v_clipped = old_val[idx] + torch.clamp(
                        new_val - old_val[idx], -self.clip_range_vf, self.clip_range_vf
                    )
                    v_loss = 0.5 * torch.max((new_val - ret[idx]) ** 2, (v_clipped - ret[idx]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((new_val - ret[idx]) ** 2).mean()

                ent_loss = -entropy.mean()

                loss = pg_loss + self.vf_coef * v_loss + self.ent_coef * ent_loss

                self.opt_policy.zero_grad()
                self.opt_value.zero_grad()
                loss.backward()
                grad_norm_last = nn.utils.clip_grad_norm_(
                    self.policy.parameters(), max_norm=self.grad_norm_clip
                ).item()
                nn.utils.clip_grad_norm_(
                    self.value_net.parameters(), max_norm=self.grad_norm_clip
                )
                self.opt_policy.step()
                self.opt_value.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(ent_loss.item())
                approx_kls.append(approx_kl)

            if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                break

        # explained variance of the critic over valid steps
        var_ret = ret.var()
        explained_variance = (
            (1.0 - (ret - old_val).var() / var_ret).item() if var_ret > 1e-8 else None
        )

        return {
            "policy_loss": sum(pg_losses) / len(pg_losses),
            "value_loss": sum(v_losses) / len(v_losses),
            "entropy": -sum(ent_losses) / len(ent_losses),
            "approx_kl": sum(approx_kls) / len(approx_kls),
            "clip_fraction": sum(clip_fracs) / len(clip_fracs),
            "grad_norm": grad_norm_last,
            "explained_variance": explained_variance,
        }

    def train(
        self,
        env_fn: Callable[[], gym.Env],
        working_dir: str | Path,

        num_timesteps: int = 1000,
        n_envs: int = 8,
        n_steps: int = 128,
        save_interval: int | None = None,
        video_interval: int | None = None,
        video_fps: int = 8,
        seed: int = 42,
    ):
        '''
        env_fn:
            a single-env constructor; vectorized here into ``n_envs`` parallel
            copies under NEXT_STEP autoreset semantics.
        n_steps:
            fixed rollout horizon per env; each update consumes ``n_steps *
            n_envs`` env interactions (dummy autoreset steps included in the
            budget but masked from the loss).
        video_interval:
            if set, record a roll-out video to TensorBoard roughly every
            ``video_interval`` env timesteps (plus one baseline at step 0).
            ``None`` disables video recording.
        '''

        envs = self.vectorize(env_fn, n_envs)

        if envs.metadata['autoreset_mode'] != AutoresetMode.NEXT_STEP:
            raise ValueError("This training requires NEXT_STEP AutoresetMode of the gym envs.")

        self.policy.train()
        self.value_net.train()

        working_dir = Path(working_dir)
        writer = SummaryWriter(working_dir)

        # baseline roll-out video before any training
        last_video_timestep = self.timesteps
        if isinstance(video_interval, int):
            record_rollout_video(
                self, env_fn, writer, self.timesteps, seed=seed, fps=video_fps
            )

        # PPO does NOT reset between rollouts: obs and prev_done carry over so
        # episodes span rollout boundaries. Reset once, here.
        obs, _ = envs.reset(seed=seed)
        prev_done = torch.zeros(n_envs, dtype=torch.bool)

        progress_bar = tqdm(range(num_timesteps), desc="PPO Training")
        new_timesteps = 0
        n_updates = 0
        last_save_timestep = self.timesteps
        start_time = time.time()

        while new_timesteps < num_timesteps:

            buffers, obs, prev_done, stats = self._collect_rollout(
                envs, obs, prev_done, n_steps, n_envs
            )

            collected = n_steps * n_envs
            new_timesteps += collected
            self.timesteps += collected
            progress_bar.update(collected)

            diag = self._update(buffers)
            n_updates += 1

            # rollout-level episode statistics (completed episodes only)
            ep_returns = stats["ep_returns"]
            ep_lengths = stats["ep_lengths"]
            mean_entropy = buffers["ent"][buffers["valid"]].mean().item()

            if ep_returns:
                ep_ret = torch.tensor(ep_returns)
                mean_reward = ep_ret.mean().item()
                return_std = ep_ret.std().item() if ep_ret.numel() > 1 else 0.0
                mean_ep_len = sum(ep_lengths) / len(ep_lengths)
                positive_reward_rate = (ep_ret > 0).float().mean().item()
            else:
                mean_reward = return_std = mean_ep_len = positive_reward_rate = None

            progress_bar.set_postfix(
                {
                    "mean_reward": "n/a" if mean_reward is None else f"{mean_reward:.4f}",
                    "approx_kl": f"{diag['approx_kl']:.4f}",
                }
            )

            time_elapsed = time.time() - start_time
            fps = int(self.timesteps / time_elapsed) if time_elapsed > 0 else 0

            self.tensorboard_write(
                writer,
                self.timesteps,

                mean_reward=mean_reward,
                policy_loss=diag["policy_loss"],
                value_loss=diag["value_loss"],
                policy_entropy=mean_entropy,
                approx_kl=diag["approx_kl"],
                clip_fraction=diag["clip_fraction"],
                clip_range=self.clip_range,
                grad_norm=diag["grad_norm"],
                explained_variance=diag["explained_variance"],
                positive_reward_rate=positive_reward_rate,
                mean_ep_len=mean_ep_len,
                return_std=return_std,
                learning_rate=self.opt_policy.param_groups[0]["lr"],
                fps=fps,
                time_elapsed=time_elapsed,
                n_updates=n_updates,
            )

            if isinstance(save_interval, int) and self.timesteps - last_save_timestep > save_interval:
                self.save_local(working_dir / f"{self.timesteps}.pt")
                last_save_timestep = self.timesteps

            if isinstance(video_interval, int) and self.timesteps - last_video_timestep > video_interval:
                record_rollout_video(
                    self, env_fn, writer, self.timesteps, seed=seed, fps=video_fps
                )
                last_video_timestep = self.timesteps

        self.save_local(working_dir / "final.pt")

    def test_reset(self):
        self.policy.eval()
        self.value_net.eval()

    def test_step(self, obs: torch.Tensor):
        with torch.no_grad():
            obs_tensor = obs[None, ...]                          # add batch dim -> (1, ...)
            act_tensor, _, _ = self.policy(obs_tensor)           # stochastic (sampled) action

        return act_tensor[0]
