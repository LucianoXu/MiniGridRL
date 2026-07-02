import torch

from minigridrl.agents.PPO import compute_gae


GAMMA = 0.9
LAM = 0.5


def _single_env(rewards, values, bootstrap, terminateds, dones):
    """Wrap 1-D per-step lists into the (T, 1) shape compute_gae expects."""
    to_col = lambda xs: torch.tensor(xs, dtype=torch.float32)[:, None]
    return compute_gae(
        to_col(rewards),
        to_col(values),
        torch.tensor([bootstrap], dtype=torch.float32),
        to_col(terminateds),
        to_col(dones),
        GAMMA,
        LAM,
    )[:, 0]


def test_terminated_step_has_no_bootstrap():
    # A terminated step's advantage is r - V(s): terminal value is 0 and the GAE
    # trace resets, so nothing downstream leaks in.
    rewards = [1.0, 5.0]
    values = [2.0, 3.0]
    terminateds = [0.0, 1.0]
    dones = [0.0, 1.0]
    adv = _single_env(rewards, values, bootstrap=99.0, terminateds=terminateds, dones=dones)

    # step 1 terminated: A = r1 - V1 = 5 - 3 = 2
    assert torch.isclose(adv[1], torch.tensor(2.0))
    # step 0 not done: delta0 = 1 + 0.9*V1 - V0 = 1 + 2.7 - 2 = 1.7; A0 = delta0 + 0.9*0.5*A1
    expected_a0 = 1.7 + GAMMA * LAM * 2.0
    assert torch.isclose(adv[0], torch.tensor(expected_a0))


def test_truncated_step_bootstraps_with_next_value():
    # A truncated step keeps its bootstrap: the *next* value is the terminal
    # observation's value (NEXT_STEP autoreset returns it as the dummy obs).
    rewards = [1.0, 5.0, 0.0]
    values = [2.0, 3.0, 7.0]        # values[2] = V(terminal obs), the dummy step
    terminateds = [0.0, 0.0, 0.0]   # NOT terminated ...
    dones = [0.0, 1.0, 0.0]         # ... but truncated at step 1
    adv = _single_env(rewards, values, bootstrap=99.0, terminateds=terminateds, dones=dones)

    # step 1 truncated: A1 = r1 + gamma*V2 - V1 = 5 + 0.9*7 - 3 = 8.3 (trace resets after)
    assert torch.isclose(adv[1], torch.tensor(8.3))


def test_dummy_step_value_does_not_leak_into_real_advantages():
    # The masked dummy step sits right after a done. Its value must not affect
    # any real step's advantage -- change it and the real advantages are stable.
    rewards = [1.0, 5.0, 0.0, 1.0]
    terminateds = [0.0, 1.0, 0.0, 0.0]
    dones = [0.0, 1.0, 0.0, 0.0]

    def adv_with_dummy_value(v_dummy):
        values = [2.0, 3.0, v_dummy, 4.0]
        return _single_env(rewards, values, bootstrap=1.0, terminateds=terminateds, dones=dones)

    a = adv_with_dummy_value(7.0)
    b = adv_with_dummy_value(-100.0)

    # real steps are 0, 1, 3 (step 2 is the dummy)
    assert torch.isclose(a[0], b[0])
    assert torch.isclose(a[1], b[1])
    assert torch.isclose(a[3], b[3])


def test_no_boundaries_matches_plain_gae():
    # With no dones, GAE reduces to the standard recursion over the whole slice.
    rewards = [1.0, 2.0, 3.0]
    values = [0.5, 0.5, 0.5]
    adv = _single_env(rewards, values, bootstrap=0.0, terminateds=[0.0] * 3, dones=[0.0] * 3)

    # backward reference
    exp = [0.0, 0.0, 0.0]
    last = 0.0
    nv = [values[1], values[2], 0.0]
    for t in reversed(range(3)):
        delta = rewards[t] + GAMMA * nv[t] - values[t]
        last = delta + GAMMA * LAM * last
        exp[t] = last
    assert torch.allclose(adv, torch.tensor(exp))
