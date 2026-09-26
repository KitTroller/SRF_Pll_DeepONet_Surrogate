"""Does the surrogate honour the +/-3 Hz limiter? Every run, every seed, gains included.

    python src/analysis/compliance.py            # ~2 min

F71 answered this on ONE run, ONE seed, and on famN -- the FIXED-gain family, because
`limiter_trace.py` cannot pass Kp/Ki -- while the deliverable is famO, a tunable-gain model.
F71 called itself "a demonstration, not a compliance certificate". This is the certificate
for the model that actually ships, and for the candidate that would replace it.

Every model on famO's 150 validation runs, each run with its OWN Kp/Ki, faults and noise
exactly as stored. The frequency the surrogate produces is dtheta/dt - omega_0 of its
recurrent rollout, by np.gradient; a sample is OUT when |f| > 1.02 L (F71's 2% margin).
The truth goes through the same finite difference: the solver clamps exactly, so whatever
the truth shows is the estimator's own floor, and the surrogate is read against it.
Reported over all samples AND over the samples where the truth is on the clamp, because
only ~3% of samples ever get near the band and an all-sample percentage hides them.
"""

# src/ on the path: these scripts live in src/analysis/ but import the pipeline
# modules (paths, PLL_Simulator, train_pll, sweep) that stay in src/. Running
# `python src/analysis/foo.py` puts src/analysis on sys.path, not src/.
# Same pattern as hpc/generate_family.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import argparse
import statistics as st

import numpy as np
import torch

from common_test import load_f32
from dataset_generator import Dataset_Creator
from paths import ROOT
from pll_infer import predict_window, _gains
from train_pll import prepare, group_split, OMEGA_BASE

LIMIT = 18.8496
T = "_W40_n{n}_W40_F4_mf503_wp0.3_s{s}sp0_L3_w128_g.pth"
MODELS = {   # name -> (checkpoints, anchor at inference)
    "deliverable (famO)":       ([f"famO{T.format(n=5000, s=s)}" for s in range(8)], False),
    "famO + anchor":            ([f"famO{T.format(n=5000, s=s)}" for s in range(8)], True),
    "famQ":                     ([f"famQ{T.format(n=10000, s=s)}" for s in range(8)], False),
    "candidate (famQ + anchor)": ([f"famQ{T.format(n=10000, s=s)}" for s in range(8)], True),
    # exp33 arm B, the flagship family (F77); the checkpoints apply the anchor themselves.
    # Flagship = seed 6 (lowest own-split record), printed on its own row above the medians.
    "flagship (famO40k, anchor trained in)":
        ([f"famO40k{T.format(n=40000, s=s).replace('_g.pth', '_ao_g.pth')}" for s in range(8)], False),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_runs", type=int, default=150)
    a = p.parse_args()

    data, meta = Dataset_Creator.load_dataset("famO_W40.npz")
    torch.set_default_dtype(torch.float32)
    prep, W, dt = prepare(data), meta["W"], meta["dt"]
    _, va = group_split(prep["run_id"], 0.15, 0)
    runs = sorted(set(prep["run_id"][va].tolist()))[:a.n_runs]

    # truth: frequency by the SAME estimator, and where the PI output u sits on the clamp
    f_true, on_clamp = [], []
    for r in runs:
        row0 = r * W
        th = torch.cat([prep["theta_abs"][row0 + k] for k in range(W)]).double().numpy()
        f_true.append(np.gradient(th, dt) - OMEGA_BASE)
        om = torch.cat([prep["target_omega"][row0 + k] for k in range(W)]).double().numpy()
        vq = torch.cat([prep["Vq"][row0 + k] for k in range(W)]).double().numpy()
        on_clamp.append(np.abs(om + float(prep["kp"][row0]) * vq) > LIMIT)
    F, C = np.concatenate(f_true), np.concatenate(on_clamp)
    print(f"{len(runs)} famO val runs, {F.size} samples; truth on the clamp: {100 * C.mean():.2f}% of samples, "
          f"in {sum(c.any() for c in on_clamp)} runs")
    print(f"TRUTH through the same estimator: outside 1.02 L on {100 * np.mean(np.abs(F) > 1.02 * LIMIT):.3f}% "
          f"of all samples, {100 * np.mean(np.abs(F[C]) > 1.02 * LIMIT):.3f}% of clamped ones  (the floor)\n")

    print(f"{'model':28s} {'seed':>4s}  {'% out (all)':>11s}  {'% out (clamped)':>15s}  {'worst |f| - L [rad/s]':>21s}")
    summary = {}
    for name, (ckpts, anchor) in MODELS.items():
        per = []
        for s, c in enumerate(ckpts):
            model, ck = load_f32(ROOT / "runs" / c)
            t = ck["t_local"]; t_ext = torch.cat([t, t[-1:] + (t[1] - t[0])])
            fp = []
            with torch.no_grad():
                for r in runs:
                    row0 = r * W
                    th0, om0, pth = prep["theta0_abs"][row0], prep["omega0"][row0], []
                    for k in range(W):
                        kp, ki = _gains(prep, row0 + k)
                        th, om = predict_window(model, ck, th0, om0, prep["Va"][row0 + k], prep["Vb"][row0 + k],
                                                prep["Vc"][row0 + k], t_ext, kp, ki, anchor_omega=anchor)
                        pth.append(th[:-1]); th0, om0 = th[-1], om[-1]
                    fp.append(np.gradient(torch.cat(pth).double().numpy(), dt) - OMEGA_BASE)
            P = np.concatenate(fp)
            out_all = 100 * np.mean(np.abs(P) > 1.02 * LIMIT)
            out_cl = 100 * np.mean(np.abs(P[C]) > 1.02 * LIMIT)
            worst = float(np.abs(P).max() - LIMIT)
            per.append((out_all, out_cl, worst))
            print(f"{name:28s} {s:4d}  {out_all:10.3f}%  {out_cl:14.3f}%  {worst:21.2f}")
        summary[name] = per
    print("\nmedian over 8 seeds [max]:")
    for name, per in summary.items():
        col = lambda i: (st.median(x[i] for x in per), max(x[i] for x in per))
        (a1, m1), (a2, m2), (a3, m3) = col(0), col(1), col(2)
        print(f"  {name:28s} out(all) {a1:.3f}% [{m1:.3f}]   out(clamped) {a2:.3f}% [{m2:.3f}]   worst overshoot {a3:.2f} [{m3:.2f}] rad/s")


if __name__ == "__main__":
    main()
