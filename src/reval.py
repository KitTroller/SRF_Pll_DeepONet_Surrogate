"""Recompute rollout metrics for existing checkpoints at a larger n_eval_runs.
No retraining. n_eval = 150 uses every validation run, removing subsampling noise.

    python reval.py sweeps_Wtest --n_eval 150
"""
import argparse, json, time
from pathlib import Path

from train_pll import load_checkpoint, prepare, group_split
from pll_infer import rollout_metrics
from paths import ROOT, sweeps as _sweeps
from dataset_generator import Dataset_Creator

p = argparse.ArgumentParser()
p.add_argument("results_dir")
p.add_argument("--n_eval", type=int, default=150)
p.add_argument("--val_frac", type=float, default=0.15)
a = p.parse_args()

cache = {}
for f in sorted(_sweeps(a.results_dir).glob("*.json")):
    r = json.loads(f.read_text())
    if r.get("status", "ok") != "ok":
        continue
    ds = r.setdefault("dataset", "pll_dataset.npz")   # records predating the field
    if ds not in cache:
        data, meta = Dataset_Creator.load_dataset(ds)
        cache[ds] = (prepare(data), meta)
    prep, meta = cache[ds]
    model, ck = load_checkpoint(ROOT / r["ckpt"])   # records store runs/<tag>.pth
    _, va = group_split(prep["run_id"], a.val_frac, r.get("split_seed", r["seed"]))
    val_runs = sorted(set(prep["run_id"][va].tolist()))
    old, t0 = r["rollout_full_rms"], time.perf_counter()
    r.update(rollout_metrics(model, ck, prep, val_runs, meta["W"], a.n_eval))
    r["n_eval_runs"] = a.n_eval
    f.write_text(json.dumps(r, indent=2))
    print(f"{r['tag']:52s} {old:.3e} -> {r['rollout_full_rms']:.3e}  "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)
