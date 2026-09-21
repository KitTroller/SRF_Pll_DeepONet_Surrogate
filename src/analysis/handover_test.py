"""The handover substitution test. Inference only, no retraining. notes.md F73.

    python src/analysis/handover_test.py 0 1 2 3        # famO L3_w128_g seeds

Seven rollouts per validation run, differing ONLY in what is handed to the next window:
    normal   predicted theta, predicted omega      the deployed rollout
    anchor   as normal, but each window's omega is shifted so it STARTS at the omega it
             was handed: omega(t) = omega0 + N(t) - N(0). The hard initial condition,
             applied after the fact to an existing checkpoint
    anchor_th    the same for theta alone: theta(t) = theta0 + w_base*t + N(t) - N(0)
    anchor_both  both at once
    true_om  predicted theta, TRUE omega
    true_th  TRUE theta,      predicted omega
    both     TRUE theta,      TRUE omega           = teacher forcing (per_window_rms)
and the step each handover makes: the next window's first omega minus the omega it was
handed. theta is HALF-anchored already -- predict_window adds theta0 exactly and the
network is trained to output 0 at t=0 -- while omega is a free output.

Anchoring is physically exact, not a patch: omega is the PI integrator's state, so it is
continuous at every instant -- through a phase jump too. Only dtheta/dt may jump.
"""

# src/ on the path: these scripts live in src/analysis/ but import the pipeline
# modules (paths, PLL_Simulator, train_pll, sweep) that stay in src/. Running
# `python src/analysis/foo.py` puts src/analysis on sys.path, not src/.
# Same pattern as hpc/generate_family.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import argparse

import torch

from dataset_generator import Dataset_Creator
from paths import ROOT
from pll_infer import predict_window, _gains
from train_pll import load_checkpoint, prepare, group_split

ARMS = ["normal", "anchor", "anchor_th", "anchor_both", "true_om", "true_th", "both"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("seeds", type=int, nargs="*", default=[0])
    p.add_argument("--dataset", default="famO_W40.npz")
    p.add_argument("--tag", default="famO_W40_n5000_W40_F4_mf503_wp0.3_s{seed}sp0_L3_w128_g")
    p.add_argument("--n_runs", type=int, default=150)
    a = p.parse_args()

    data, meta = Dataset_Creator.load_dataset(a.dataset)
    prep, W = prepare(data), meta["W"]
    _, va = group_split(prep["run_id"], 0.15, 0)
    val_runs = sorted(set(prep["run_id"][va].tolist()))[:a.n_runs]

    for seed in a.seeds:
        model, ck = load_checkpoint(ROOT / "runs" / f"{a.tag.format(seed=seed)}.pth")
        t = ck["t_local"]
        t_ext = torch.cat([t, t[-1:] + (t[1] - t[0])])
        th_err = {k: [] for k in ARMS}; om_err = {k: [] for k in ARMS}
        steps = []
        with torch.no_grad():
            for r in val_runs:
                row0 = r * W
                state = {k: (prep["theta0_abs"][row0], prep["omega0"][row0]) for k in ARMS}
                pth = {k: [] for k in ARMS}; pom = {k: [] for k in ARMS}
                for w in range(W):
                    kp, ki = _gains(prep, row0 + w)
                    V = (prep["Va"][row0 + w], prep["Vb"][row0 + w], prep["Vc"][row0 + w])
                    for k in ARMS:
                        th0, om0 = state[k]
                        th, om = predict_window(model, ck, th0, om0, *V, t_ext, kp, ki)
                        if k in ("anchor", "anchor_both"):
                            om = om + (om0 - om[0])
                        if k in ("anchor_th", "anchor_both"):
                            th = th + (th0 - th[0])
                        if k == "normal" and w > 0:
                            steps.append(float(om[0] - om0))
                        pth[k].append(th[:-1]); pom[k].append(om[:-1])
                        if w + 1 < W:
                            tt, to = prep["theta0_abs"][row0 + w + 1], prep["omega0"][row0 + w + 1]
                            state[k] = (tt if k in ("true_th", "both") else th[-1],
                                        to if k in ("true_om", "both") else om[-1])
                tth = torch.cat([prep["theta_abs"][row0 + w] for w in range(W)])
                tom = torch.cat([prep["target_omega"][row0 + w] for w in range(W)])
                for k in ARMS:
                    th_err[k].append(torch.cat(pth[k]) - tth)
                    om_err[k].append(torch.cat(pom[k]) - tom)
        rms = lambda es: float(torch.stack([e.pow(2).mean().sqrt() for e in es]).mean())
        mx = lambda es: float(torch.stack([e.abs().max() for e in es]).max())
        print(f"\n== seed {seed}  ({len(val_runs)} val runs, {W} windows)")
        print(f"{'arm':8s} {'theta RMS':>10s} {'theta max':>10s} {'omega RMS':>10s}")
        for k in ARMS:
            print(f"{k:8s} {rms(th_err[k]):10.3e} {mx(th_err[k]):10.3e} {rms(om_err[k]):10.3e}")
        s = torch.tensor(steps).abs()
        print(f"omega step at the handover: median {s.median():.2e}  p99 {s.quantile(.99):.2e}  "
              f"max {s.max():.2e} rad/s")


if __name__ == "__main__":
    main()
