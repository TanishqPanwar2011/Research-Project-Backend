"""Layer 2: CNN-LSTM load predictor (PyTorch)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn

from .layer2_data import SEQ, prepare


class CNNLSTM(nn.Module):
    """1D CNN picks up short-range patterns inside the 8-step window, the LSTM models the sequence."""

    def __init__(self, n_features: int, channels: int = 32, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, channels, 3, padding=1), nn.ReLU(),
            nn.Conv1d(channels, channels, 3, padding=1), nn.ReLU(),
        )
        self.lstm = nn.LSTM(channels, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):                                  # x: (B, T, F)
        h = self.cnn(x.transpose(1, 2)).transpose(1, 2)    # (B, T, C)
        o, _ = self.lstm(h)
        return self.head(o[:, -1]).squeeze(-1)             # last time step -> one number


def _predict(model, x, device, bs=4096) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            out.append(model(torch.from_numpy(x[i:i + bs]).to(device)).cpu().numpy())
    return np.concatenate(out)


def run_layer2(df: pd.DataFrame, feats: list[str], epochs=15, batch=512, lr=1e-3, patience=4,
               delta=True, seed=0, pool_size=400, log=print) -> tuple[dict, dict]:
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = prepare(df, feats, SEQ, delta, seed=seed)
    tr, va, te = d["train"], d["val"], d["test"]

    model = CNNLSTM(len(feats)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    xt, zt = torch.from_numpy(tr["x"]), torch.from_numpy(tr["z"].astype(np.float32))
    best, best_state, bad = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), batch):
            j = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(model(xt[j].to(device)), zt[j].to(device))
            loss.backward()
            opt.step()
        vz = _predict(model, va["x"], device)
        vloss = float(np.mean((vz - va["z"]) ** 2))
        log(f"    epoch {ep:2d}  val loss {vloss:.4f}")
        if vloss < best - 1e-5:
            best, bad = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)

    pred = _predict(model, te["x"], device) * d["sd"] + d["mu"] + te["base"]
    pred = np.clip(pred, 0, None)
    y, cur = te["y"], te["cur"]
    metrics = {
        "mse": float(mean_squared_error(y, pred)), "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
        "persistenceR2": float(r2_score(y, cur)), "persistenceMSE": float(mean_squared_error(y, cur)),
    }

    # Cluster-mean actual vs predicted per 5-minute window of the test period (for the UI chart)
    g = pd.DataFrame({"b": te["bucket"], "a": y, "p": pred}).groupby("b").mean().sort_index()
    # A fixed pool of real test samples (same indices for every feature set): used as tasks by the scheduler
    pick = np.random.default_rng(123).choice(len(y), min(pool_size, len(y)), replace=False)
    result = {
        "features": feats, "delta": delta, "seq": SEQ, "metrics": metrics,
        "series": {"actual": g["a"].round(6).tolist(), "predicted": g["p"].round(6).tolist()},
        "pool": {"actual": y[pick].round(6).tolist(), "predicted": pred[pick].round(6).tolist(),
                 "scale": float(np.percentile(y, 99)) or 1.0},
        "rows": {"train": len(tr["y"]), "val": len(va["y"]), "test": len(te["y"])},
        "epochs_run": ep, "seconds": round(time.time() - t0, 1),
    }
    return result, best_state
