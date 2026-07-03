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


def test_update_reduces_inverse_loss_on_fixed_batch():
    # The inverse model's target is the fixed action labels, so encoder+inverse
    # can overfit a fixed batch and the inverse loss reliably drops. (Forward
    # loss is NOT a valid monotonicity target: the jointly-trained encoder moves
    # phi(s'), so the forward regression target shifts underneath it.)
    icm = ICM(beta=0.5, lr=1e-3, feature_dim=32, hidden_dims=(32,))
    obs, action, next_obs = _batch()
    before = icm.update(obs, action, next_obs)["icm_inverse_loss"]
    after = before
    for _ in range(100):
        after = icm.update(obs, action, next_obs)["icm_inverse_loss"]
    assert after < before
