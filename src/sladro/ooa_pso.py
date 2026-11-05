"""OOA-PSO: orthogonal-array exploration + binary particle swarm exploitation.

Each iteration does what Algorithm 1 in the paper describes:
  Step 1 (exploration): an orthogonal array mixes the best subset found so far with another
                        particle's best subset, testing the mixes in a balanced way.
  Step 2 (exploitation): standard PSO velocity/position update, with a sigmoid turning each
                        velocity into the probability that a feature is kept (binary PSO).
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .orthogonal import orthogonal_array


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class OOAPSOResult:
    mask: np.ndarray  # 1 = feature kept
    fitness: float
    history: list = field(default_factory=list)  # best fitness after each iteration
    n_evals: int = 0  # distinct subsets actually scored


class OOAPSO:
    def __init__(self, fitness: Callable[[np.ndarray], float], n_features: int, n_iter: int = 30,
                 w: float = 0.7, c1: float = 1.5, c2: float = 1.5, v_max: float = 4.0,
                 patience: int = 8, eps: float = 1e-4, seed: int = 0):
        self.fitness, self.n = fitness, n_features
        self.n_iter, self.w, self.c1, self.c2, self.v_max = n_iter, w, c1, c2, v_max
        self.patience, self.eps = patience, eps
        self.rng = np.random.default_rng(seed)
        self._cache: dict = {}

    def _eval(self, mask) -> float:
        key = tuple(int(b) for b in mask)
        if key not in self._cache:  # the same subset is never scored twice
            self._cache[key] = float(self.fitness(np.array(key))) if any(key) else -1.0
        return self._cache[key]

    def run(self) -> OOAPSOResult:
        oa = orthogonal_array(self.n)
        p = oa.shape[0]
        x = oa.copy()  # the first swarm IS the orthogonal array
        v = np.zeros((p, self.n))
        pbest = x.copy()
        pfit = np.array([self._eval(r) for r in x])
        g = int(pfit.argmax())
        gbest, gfit = pbest[g].copy(), pfit[g]
        history, stall = [gfit], 0

        for _ in range(self.n_iter):
            prev = gfit
            # Step 1: orthogonal exploration around the global best
            partner = pbest[self.rng.integers(p)]
            for row in np.where(oa == 0, gbest, partner):
                f = self._eval(row)
                if f > gfit:
                    gfit, gbest = f, row.copy()
            # Step 2: binary PSO
            r1, r2 = self.rng.random((p, self.n)), self.rng.random((p, self.n))
            v = self.w * v + self.c1 * r1 * (pbest - x) + self.c2 * r2 * (gbest - x)
            v = np.clip(v, -self.v_max, self.v_max)
            x = (self.rng.random((p, self.n)) < _sigmoid(v)).astype(int)
            for i in range(p):
                f = self._eval(x[i])
                if f > pfit[i]:
                    pfit[i], pbest[i] = f, x[i].copy()
                if f > gfit:
                    gfit, gbest = f, x[i].copy()
            history.append(gfit)
            stall = stall + 1 if gfit - prev < self.eps else 0
            if stall >= self.patience:  # convergence threshold (epsilon) from the paper
                break
        return OOAPSOResult(gbest, float(gfit), [float(h) for h in history], len(self._cache))
