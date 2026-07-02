from pathlib import Path
import time

import gymnasium as gym
from gymnasium.vector import AutoresetMode
from gymnasium.wrappers.vector import NumpyToTorch

import torch
from torch import nn
from torch.optim import Adam

from .interface import RLAgent
from ..models.interface import RLModule

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm


class REINFORCE(RLAgent):

    def __init__(
        self,
        policy: RLModule,

        
        lr = 0.001,
        betas = (0.9, 0.999),
        eps = 1e-8,
        weight_decay = 0.,
        grad_norm_clip = 1.0,
    ):
        super().__init__()

        self.policy = policy
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.grad_norm_clip = grad_norm_clip
        self.timesteps = 0

        self.opt = Adam(
            policy.parameters(),
            lr = lr,
            betas = betas,
            eps = eps,
            weight_decay= weight_decay
        )

    def save_local(self, pt_path: str | Path):
        '''
        Save the full policy module (architecture + weights), the optimizer
        state, and the hyperparameters into pt_path.
        '''
        pt_path = Path(pt_path)
        pt_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "policy": self.policy,               # full nn.Module (pickled)
                "optimizer_state": self.opt.state_dict(),
                "timesteps": self.timesteps,
                "hyperparams": {
                    "lr": self.lr,
                    "betas": self.betas,
                    "eps": self.eps,
                    "weight_decay": self.weight_decay,
                    "grad_norm_clip": self.grad_norm_clip
                },
            },
            pt_path,
        )

    @classmethod
    def load_local(cls, pt_path: str | Path):
        '''
        Rebuild a REINFORCE agent from a checkpoint written by `save_local`.
        '''
        pt_path = Path(pt_path)

        # weights_only=False: the checkpoint contains a pickled nn.Module, not
        # just tensors. Only load checkpoints you trust.
        ckpt = torch.load(
            pt_path,
            map_location="cpu",
            weights_only=False,
        )

        agent = cls(policy=ckpt["policy"], **ckpt["hyperparams"])
        agent.opt.load_state_dict(ckpt["optimizer_state"])
        agent.timesteps = ckpt["timesteps"]
        return agent

    def train(
        self,
        envs: gym.vector.AsyncVectorEnv,
        working_dir: str | Path,

        num_timesteps: int = 1000,
        save_interval: int | None = None,
        seed: int = 42,
    ):
        '''
        policy:
            input: batched observation, torch.Tensor type
            output: tuple of batched actions, log_probabilities, torch.Tensor type each.
        '''

        envs = NumpyToTorch(envs)
        n_envs = envs.num_envs

        if envs.metadata['autoreset_mode'] != AutoresetMode.NEXT_STEP:
            raise ValueError("This training requires NEXT_STEP AutoresetMode of the gym envs.")
        
        self.policy.train()

        # create the summary writer
        working_dir = Path(working_dir)
        writer = SummaryWriter(working_dir)
        
        obs, info = envs.reset(seed=seed)
        

        progress_bar = tqdm(range(num_timesteps), desc="RL Training")
        new_timesteps = 0
        n_updates = 0
        start_time = time.time()

        while new_timesteps < num_timesteps:
            
            # build the clean buffer
            
            obs, _ = envs.reset()

            # assume every episode will have at_least one action step
            ongoing_mask = torch.tensor([True]*n_envs)

            raw_reward = torch.zeros(n_envs, 0)
            reward_to_go = torch.zeros(n_envs, 0)
            lp_trace = torch.zeros(n_envs, 0)
            ent_trace = torch.zeros(n_envs, 0)
            step_mask = torch.ones(n_envs, 0)

            last_save_timestep = self.timesteps

            # finish one batch of episode
            while bool(ongoing_mask.any()):

                # assume every episode will have at_least one action step
                step_mask = torch.concat([step_mask, ongoing_mask.float()[:, None]], dim = 1)

                selected_obs = obs[ongoing_mask]

                selected_acts, selected_lp, selected_ent = self.policy(selected_obs)

                # build the value with placeholders for them
                acts = torch.as_tensor(envs.action_space.sample())
                acts[ongoing_mask] = selected_acts

                lp = torch.zeros(n_envs)
                lp[ongoing_mask] = selected_lp

                ent = torch.zeros(n_envs)
                ent[ongoing_mask] = selected_ent.detach()   # logging only, keep out of the graph

                obs, rewards, completed, truncated, _ = envs.step(acts)

                r = (rewards * ongoing_mask.float())[:, None]
                raw_reward = torch.concat([raw_reward, r], dim=1)
                reward_to_go += r
                reward_to_go = torch.concat([reward_to_go, r], dim=1)

                lp_trace = torch.concat([lp_trace, lp[:, None]], dim=1)
                ent_trace = torch.concat([ent_trace, ent[:, None]], dim=1)

                # upate ongoing_mask
                finished_mask = ongoing_mask.logical_not().logical_or(
                    torch.logical_or(completed, truncated)
                )
                ongoing_mask = finished_mask.logical_not()

            # update timesteps
            batch_timesteps = step_mask.numel()
            new_timesteps += batch_timesteps
            progress_bar.update(batch_timesteps)
            self.timesteps += batch_timesteps

            # episode-level statistics
            episode_return = raw_reward.sum(dim=1)                 # (n_envs,)
            episode_length = step_mask.sum(dim=1)                  # (n_envs,)

            mean_reward = episode_return.mean()
            return_std = episode_return.std()
            mean_ep_len = episode_length.mean()
            # mean policy entropy over valid (ongoing) steps only
            mean_entropy = ent_trace[step_mask.bool()].mean()

            positive_reward_rate = (episode_return > 0).float().mean()

            # build the average baseline and advantanges
            baseline = reward_to_go.sum(dim=0) / step_mask.sum(dim=0).clamp(min=1)
            advantages = (reward_to_go - baseline[None, :]).detach()

            # explained variance of the baseline as a predictor of reward-to-go
            # (valid steps only): 1 - Var(returns - baseline) / Var(returns).
            # None when returns have ~zero variance (undefined).
            valid = step_mask.bool()
            var_returns = reward_to_go[valid].var()
            explained_variance = (
                (1 - advantages[valid].var() / var_returns).item()
                if var_returns > 1e-8 else None
            )

            # build the surrogate loss
            J = - (advantages * lp_trace).sum(dim=1).mean()

            progress_bar.set_postfix(
                {
                    "mean_reward": f"{mean_reward.item():.4f}",
                    "pos_reward_rate": f"{positive_reward_rate.item():.2f}",
                }
            )

            self.opt.zero_grad()
            J.backward()

            grad_norm = nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                max_norm = self.grad_norm_clip
            )

            self.opt.step()
            n_updates += 1

            # timing / throughput
            time_elapsed = time.time() - start_time
            fps = int(self.timesteps / time_elapsed) if time_elapsed > 0 else 0

            # write to tensorboard
            self.tensorboard_write(
                writer,
                self.timesteps,

                # --- applicable to REINFORCE ---
                mean_reward=mean_reward.item(),
                surrogate_loss=J.item(),
                grad_norm=grad_norm.item(),
                policy_entropy=mean_entropy.item(),
                positive_reward_rate=positive_reward_rate.item(),
                mean_ep_len=mean_ep_len.item(),
                return_std=return_std.item(),
                explained_variance=explained_variance,
                learning_rate=self.opt.param_groups[0]["lr"],
                fps=fps,
                time_elapsed=time_elapsed,
                n_updates=n_updates,

                # --- N/A for REINFORCE (no critic / not PPO) -> None, skipped ---
                approx_kl=None,       # no old/new policy ratio in vanilla REINFORCE
                value_loss=None,      # no value network
                clip_fraction=None,   # PPO-specific
                clip_range=None,      # PPO-specific
            )

            # save the ckpt
            if isinstance(save_interval, int) and self.timesteps - last_save_timestep > save_interval:
                self.save_local(working_dir / f"{self.timesteps}.pt")
                last_save_timestep = self.timesteps
                

        # save the final ckpt
        self.save_local(working_dir / "final.pt")



    def test_reset(self):
        self.policy.eval()
    
    def test_step(self, obs: torch.Tensor):
        with torch.no_grad():
            obs_tensor = obs[None, ...]                          # add batch dim -> (1, ...)

            act_tensor, _, _ = self.policy(obs_tensor)           # stochastic (sampled) action

        return act_tensor[0]