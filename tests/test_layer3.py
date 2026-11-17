import numpy as np
import pytest

from sladro.layer3_env import MAX_VMS, N_FEATURES, DQNPolicy, SchedEnv, run_schedule, vm_features, vm_specs


def test_features_shape_and_finite_for_any_vm_count():
    for v in (2, 7, MAX_VMS):
        mips, power = vm_specs(v)
        x = vm_features(np.random.default_rng(0).random(v) * 10, 3000.0, mips, power)
        assert x.shape == (v, N_FEATURES) and np.isfinite(x).all()


def test_env_charges_true_length_and_rewards_makespan_growth():
    env = SchedEnv([2000, 2000], [100, 100], 2)  # estimate is wrong on purpose: true length is charged
    r1 = env.step(0)
    assert env.f[0] == pytest.approx(2000 / 1000) and r1 < 0
    r2 = env.step(1)  # lands on the idle VM: makespan does not grow
    assert env.done and r2 > r1


def test_round_robin_and_earliest_finish_use_all_vms():
    rng = np.random.default_rng(0)
    t = 800 + 6000 * rng.random(100)
    for mode in ("rr", "earliest"):
        assert (run_schedule(t, t, 5, mode) > 0).all()
    assert run_schedule(t, t, 5, "earliest").max() <= run_schedule(t, t, 5, "rr").max()


def test_policy_scores_every_vm_and_acts_in_range():
    rng = np.random.default_rng(0)
    pol = DQNPolicy([(rng.normal(size=(6, 8)).astype(np.float32), np.zeros(8, np.float32)),
                     (rng.normal(size=(8, 1)).astype(np.float32), np.zeros(1, np.float32))])
    x = vm_features(np.zeros(9), 3000.0, *vm_specs(9))
    assert pol.q(x).shape == (9,) and 0 <= pol.act(x) < 9
    assert run_schedule([3000.0] * 20, [3000.0] * 20, 9, "dqn", pol).shape == (9,)


def test_torch_export_matches_numpy_inference():
    torch = pytest.importorskip("torch")
    from sladro.layer3 import QNet, export_policy
    net = QNet()
    x = np.random.default_rng(1).normal(size=(10, N_FEATURES)).astype(np.float32)
    pol = DQNPolicy.from_dict(export_policy(net))
    assert np.allclose(pol.q(x), net(torch.from_numpy(x)).detach().numpy(), atol=1e-5)