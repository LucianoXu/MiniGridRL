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
    after = before
    for _ in range(50):
        lpm.compute_intrinsic(obs, action, next_obs)
        after = lpm.update(obs, action, next_obs)["lpm_dynamics_loss"]
    assert after < before


def test_error_model_is_fixed_encoder_frozen():
    lpm = LPM(feature_dim=16, hidden_dims=(16,))
    assert all(not p.requires_grad for p in lpm.psi.parameters())
