"""Layer 1: feature selection with OOA-PSO, scored on a time-ordered train/validation/test split."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from .data2011 import FEATURES, TARGET
from .ooa_pso import OOAPSO


class SubsetFitness:
    """fitness(mask) = validation R2 of a Ridge model on the chosen features - beta * (fraction of features used).

    The penalty is the paper's accuracy-versus-cost trade-off (alpha and beta in Eq. 30), simplified.
    """

    def __init__(self, x_train, y_train, x_val, y_val, beta=0.05, ridge_alpha=1.0):
        self.scaler = StandardScaler().fit(x_train)
        self.xt, self.yt = self.scaler.transform(x_train), y_train
        self.xv, self.yv = self.scaler.transform(x_val), y_val
        self.beta, self.ridge_alpha, self.n = beta, ridge_alpha, x_train.shape[1]

    def _r2(self, mask, xa, ya, xb, yb) -> float:
        idx = np.flatnonzero(mask)
        model = Ridge(alpha=self.ridge_alpha).fit(xa[:, idx], ya)
        return float(r2_score(yb, model.predict(xb[:, idx])))

    def __call__(self, mask) -> float:
        return self._r2(mask, self.xt, self.yt, self.xv, self.yv) - self.beta * mask.sum() / self.n

    def r2_on(self, mask, x_raw, y) -> float:
        """Train on the training set, score on any other raw (unscaled) data."""
        return self._r2(mask, self.xt, self.yt, self.scaler.transform(x_raw), y)


def time_split(df: pd.DataFrame, fracs=(0.6, 0.2)):
    """Split by TIME (never randomly), so the model is never tested on the past it was trained on."""
    b = np.sort(df["bucket"].unique())
    if len(b) < 5:
        raise ValueError("Need at least 5 time windows. Download more part files.")
    i1, i2 = int(len(b) * fracs[0]), int(len(b) * (fracs[0] + fracs[1]))
    return (df[df["bucket"] < b[i1]], df[(df["bucket"] >= b[i1]) & (df["bucket"] < b[i2])], df[df["bucket"] >= b[i2]])


def run_layer1(df: pd.DataFrame, n_iter=30, beta=0.05, seed=0, max_train_rows=100_000, max_eval_rows=50_000) -> dict:
    t0 = time.time()
    tr, va, te = time_split(df)
    pick = lambda d, n: d.sample(n=min(n, len(d)), random_state=seed)  # subsample: keeps scoring fast
    tr, va, te = pick(tr, max_train_rows), pick(va, max_eval_rows), pick(te, max_eval_rows)
    arr = lambda d: d[FEATURES].to_numpy(dtype=float)
    fit = SubsetFitness(arr(tr), tr[TARGET].to_numpy(), arr(va), va[TARGET].to_numpy(), beta)

    res = OOAPSO(fit, len(FEATURES), n_iter=n_iter, seed=seed).run()
    full = np.ones(len(FEATURES), dtype=int)
    xte, yte = arr(te), te[TARGET].to_numpy()
    return {
        "all_features": FEATURES,
        "selected": [f for f, m in zip(FEATURES, res.mask) if m],
        "mask": [int(m) for m in res.mask],
        "fitness": res.fitness,
        "history": res.history,
        "n_evals": res.n_evals,
        "r2": {
            "val_all": fit.r2_on(full, arr(va), va[TARGET].to_numpy()),
            "val_selected": fit.r2_on(res.mask, arr(va), va[TARGET].to_numpy()),
            "test_all": fit.r2_on(full, xte, yte),
            "test_selected": fit.r2_on(res.mask, xte, yte),
            "test_persistence": float(r2_score(yte, te["cpu_rate"])),  # naive: "next = current"
        },
        "rows": {"train": len(tr), "val": len(va), "test": len(te)},
        "settings": {"n_iter": n_iter, "beta": beta, "seed": seed},
        "seconds": round(time.time() - t0, 1),
    }
