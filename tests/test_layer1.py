import numpy as np

from sladro.layer1 import SubsetFitness
from sladro.ooa_pso import OOAPSO
from sladro.orthogonal import orthogonal_array


def test_orthogonal_array_is_balanced_and_pairwise_even():
    oa = orthogonal_array(11)
    n = oa.shape[0]
    assert oa.shape == (16, 11)
    assert (oa.sum(axis=0) == n // 2).all()
    for i in range(11):
        for j in range(i + 1, 11):
            for a in (0, 1):
                for b in (0, 1):
                    assert ((oa[:, i] == a) & (oa[:, j] == b)).sum() == n // 4


def test_ooa_pso_finds_planted_features():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4000, 8))
    y = 2 * x[:, 0] - 1.5 * x[:, 3] + 0.1 * rng.normal(size=4000)
    fit = SubsetFitness(x[:2500], y[:2500], x[2500:], y[2500:], beta=0.05)
    res = OOAPSO(fit, 8, n_iter=15, seed=0).run()
    assert res.mask[0] == 1 and res.mask[3] == 1
    assert res.mask.sum() <= 4
