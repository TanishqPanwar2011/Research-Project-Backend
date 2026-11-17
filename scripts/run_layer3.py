"""Train the Layer 3 DQN scheduler from the Layer 2 artifacts and save artifacts/layer3.json for the API.

Needs no trace download: the tasks are the real test samples that Layer 2 already saved.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sladro.layer3 import evaluate, train_dqn  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--art", default="artifacts")
ap.add_argument("--episodes", type=int, default=800)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

art = Path(a.art)
l2_all = json.loads((art / "layer2_all.json").read_text())
l2_sel = json.loads((art / "layer2_selected.json").read_text())
pool = l2_all["pool"]
preds = {"all": pool["predicted"], "selected": l2_sel["pool"]["predicted"]}

print(f"Training DQN for {a.episodes} episodes on {int(len(pool['actual']) * 0.75)} real tasks...")
policy, info = train_dqn(pool["actual"], preds, pool["scale"], episodes=a.episodes, seed=a.seed)
held = info.pop("held_out")
ev = evaluate(policy, held, pool["scale"])

(art / "layer3.json").write_text(json.dumps({
    "policy": policy, "curve": info["curve"], "eval": ev,
    "settings": {"episodes": a.episodes, "seed": a.seed, "seconds": info["seconds"]},
}))

print("\nMean makespan (s) on held-out tasks, 200 tasks per run (lower is better)")
print(f"  {'Layer 2':9s}{'VMs':>4s}{'RoundRobin':>12s}{'Earliest':>10s}{'DQN':>9s}{'DQN vs RR':>11s}{'DQN vs EF':>11s}")
for src, by_vm in ev.items():
    for v, m in by_vm.items():
        print(f"  {src:9s}{v:>4s}{m['rr']:12.2f}{m['earliest']:10.2f}{m['dqn']:9.2f}"
              f"{(1 - m['dqn'] / m['rr']) * 100:10.1f}%{(1 - m['dqn'] / m['earliest']) * 100:10.1f}%")
print(f"\nTrained in {info['seconds']}s. Saved {art / 'layer3.json'} (commit this)")