"""Deployed metrics split by fault type.

    python src/fault_split.py runs/famD_W40_n5000_W40_F4_mf503_wp0.3_s0sp0.pth

A famD record's `rollout_full_rms` averages clean and faulted validation runs together,
so it is comparable to nothing -- not to famB, not to the paper, not to any number in
notes.md. This reruns `rollout_metrics` on each `fault_kind` subset separately.

The clean subset is the one to check first: it should land near famB's numbers. If it
does not, the fault runs are degrading clean performance (catastrophic interference),
which is a result in itself and the argument for weighting the loss by fault type.
"""
import argparse
import json

import numpy as np

from dataset_generator import Dataset_Creator
from paths import ROOT
from pll_infer import rollout_metrics
from train_pll import group_split, load_checkpoint, prepare

NAMES = {0: "clean", 1: "sag", 2: "phase_jump"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--dataset", default=None, help="default: read from the sweep record")
    p.add_argument("--n_eval", type=int, default=150)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--split_seed", type=int, default=0)
    a = p.parse_args()

    dataset = a.dataset
    if dataset is None:                       # find the record that names this ckpt
        for f in (ROOT / "Hyperparameter_sweep").rglob("*.json"):
            r = json.loads(f.read_text())
            if r.get("ckpt", "").endswith(a.ckpt.split("/")[-1]):
                dataset = r["dataset"]
                break
        if dataset is None:
            raise SystemExit("no record names this checkpoint; pass --dataset")

    data, meta = Dataset_Creator.load_dataset(dataset)
    if "fault_kind" not in data:
        raise SystemExit(f"{dataset} has no fault_kind -- it predates disturbances")
    prep = prepare(data)
    W = meta["W"]
    model, ck = load_checkpoint(ROOT / a.ckpt if not a.ckpt.startswith("/") else a.ckpt)

    _, va = group_split(prep["run_id"], a.val_frac, a.split_seed)
    val_runs = sorted(set(prep["run_id"][va].tolist()))
    kind = data["fault_kind"][::W].numpy()               # (n_runs,), per-run value

    print(f"{dataset}  W={W}  {len(val_runs)} validation runs  n_eval={a.n_eval}\n")
    print(f"  {'subset':11s} {'n':>4s} {'roll_rms':>11s} {'roll_med':>11s} "
          f"{'per_win':>11s} {'comp':>6s}")
    for k, name in NAMES.items():
        subset = [r for r in val_runs if kind[r] == k]
        if not subset:
            continue
        m = rollout_metrics(model, ck, prep, subset, W, a.n_eval)
        print(f"  {name:11s} {min(len(subset), a.n_eval):4d} "
              f"{m['rollout_full_rms']:11.4e} {m['rollout_full_med']:11.4e} "
              f"{m['per_window_rms']:11.4e} {m['compounding']:6.2f}")

    # the number the record itself reported, for reference: all runs mixed
    m = rollout_metrics(model, ck, prep, val_runs, W, a.n_eval)
    print(f"  {'ALL (mixed)':11s} {min(len(val_runs), a.n_eval):4d} "
          f"{m['rollout_full_rms']:11.4e} {m['rollout_full_med']:11.4e} "
          f"{m['per_window_rms']:11.4e} {m['compounding']:6.2f}   <- what the record says")


if __name__ == "__main__":
    main()
