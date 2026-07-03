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
