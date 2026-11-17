"""SLADRO backend: serves the same RunResult JSON as the UI's placeholder pipeline.

Layers 1 and 2 come from pre-computed artifacts (trained offline on the Google 2011 trace), so this
service needs no torch and starts instantly. Layer 3 is still the earliest-finish placeholder.
Run:  uvicorn --app-dir src sladro.api:app --port 8000
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ART = Path(__file__).resolve().parents[2] / "artifacts"
MIPS = [1000, 1500, 2000, 3000, 4000]
POWER = [60, 80, 110, 150, 200]

app = FastAPI(title="SLADRO backend")
_cache: dict = {}


def art(name: str) -> dict:
    if name not in _cache:
        p = ART / name
        if not p.exists():
            raise HTTPException(503, f"Missing {p.name}. Run scripts/run_layer2.py first.")
        _cache[name] = json.loads(p.read_text())
    return _cache[name]


class Config(BaseModel):
    featureSelection: bool = True
    predictor: bool = True
    scheduler: bool = True
    seed: int = 42
    numTasks: int = 200
    numVMs: int = 5


def schedule(true_len, est_len, vms: int, smart: bool) -> list[float]:
    f = [0.0] * vms
    for i, ln in enumerate(true_len):
        a = i % vms  # Round Robin
        if smart:  # placeholder for the DQN: pick the VM that finishes the ESTIMATED task earliest
            a = min(range(vms), key=lambda m: f[m] + est_len[i] / MIPS[m % 5])
        f[a] += ln / MIPS[a % 5]
    return f


@app.get("/health")
def health():
    return {"ok": True, "artifacts": sorted(p.name for p in ART.glob("*.json"))}


@app.post("/run")
def run(c: Config):
    l1 = art("layer1.json")
    all_feats = l1["all_features"]
    selected = l1["selected"] if c.featureSelection else all_feats
    l2 = art("layer2_selected.json" if c.featureSelection else "layer2_all.json")
    base_pool = art("layer2_all.json")["pool"]  # same real test samples for every combination

    vms, n = max(2, min(20, c.numVMs)), max(10, min(400, c.numTasks))
    rng = np.random.default_rng(c.seed)
    pick = rng.choice(len(base_pool["actual"]), n, replace=n > len(base_pool["actual"]))
    scale = base_pool["scale"]
    to_len = lambda v: 800 + 6000 * np.clip(np.asarray(v) / scale, 0, 1)
    true_len = to_len(np.asarray(base_pool["actual"])[pick])
    est_len = to_len(np.asarray(l2["pool"]["predicted"])[pick]) if c.predictor else np.full(n, true_len.mean())

    f = np.array(schedule(true_len.tolist(), est_len.tolist(), vms, c.scheduler))
    mk = float(f.max())
    u = f / mk
    energy = sum(POWER[m % 5] * b + 0.4 * POWER[m % 5] * (mk - b) for m, b in enumerate(f)) / 1000
    return {
        "config": c.model_dump(),
        "allFeatures": all_feats,
        "selectedFeatures": selected,
        "predictor": l2["metrics"] if c.predictor else None,
        "schedulerName": "DQN (placeholder: earliest-finish)" if c.scheduler else "Round Robin",
        "metrics": {"makespan": mk, "throughput": n / mk, "utilizationSD": float(u.std()), "energyKJ": float(energy)},
        "series": {"actual": l2["series"]["actual"][-80:],
                   "predicted": l2["series"]["predicted"][-80:] if c.predictor else None},
        "vmBusy": f.tolist(),
    }
