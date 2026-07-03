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
