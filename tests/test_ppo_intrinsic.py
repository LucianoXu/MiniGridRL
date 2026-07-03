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
