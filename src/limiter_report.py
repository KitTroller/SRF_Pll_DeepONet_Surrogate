"""What did the Siemens frequency limiter cost? graphs/24.

    python src/limiter_report.py            # ~4 min

famN and famR are the same family generated twice from --lhs_seed 21, with and without
the limiter. The limiter acts inside the integrator, AFTER _grid_phases, so the two share
Va/Vb/Vc bit-exactly and differ only in the PLL trajectory.

WHY EACH MODEL IS SCORED ON ITS OWN PHYSICS, and why that is not the F59 mistake.
F59 scored models on different validation SPLITS of the SAME physics, so a difficulty
difference masqueraded as a model difference. Here the physics genuinely differs by
design: famN's ground truth is the limited ODE and famR's is the unlimited one. The
question is not "which model is better on one task" but "is the limited problem harder to
learn", so each model is measured against the trajectory it is supposed to reproduce,
from identical initial conditions.

THE HEADLINE NUMBER IS USELESS ON ITS OWN. Only ~2.6% of samples touch the clamp, so an
aggregate theta RMS averages the limiter away and famN and famR come out identical
whatever the truth is. Every window is therefore flagged by whether the TRUTH saturated
in it -- max|omega + Kp*Vq| > L -- and reported separately, the way fault_split.py
splits by fault kind.
"""
import argparse
import glob
import statistics as st

import numpy as np
import torch

import PLL_Simulator as PS
from common_test import load_f32, rollout
from paths import graphs as _graphs

DT, N, N_RUNS = 100e-6, 5000, 12
LIMIT = 18.8496          # 2*pi*3 rad/s


def truth(limit, n_runs=N_RUNS, seed=0):
    """Identical ICs whatever `limit` is: the seed is consumed in a fixed draw order and
    the limiter only acts inside simulate_batch, after the grid waveform is built."""
    PS.pll_constants.freq_limit = limit
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    sim = PS.PLLSimulator(dt=DT)
    sim.N, sim.n_runs = N, n_runs
    sim.t = (torch.arange(N) * DT).reshape(1, N)
    ga = (torch.rand(n_runs, 1) * 2 - 1) * torch.pi
    fo = (torch.rand(n_runs, 1) * 2 - 1) * 0.2
    ao = (torch.rand(n_runs, 1) * 2 - 1) * 0.05
    th0 = ga.squeeze(-1) + (torch.rand(n_runs) * 2 - 1) * 0.5 * torch.pi
    th0 = (th0 + torch.pi) % (2 * torch.pi) - torch.pi
    om0 = (torch.rand(n_runs) * 2 - 1) * 20.0
    Va, Vb, Vc = sim._grid_phases(fo, ao, ga)
    with torch.no_grad():
        th, om, _, Vq = sim.simulate_batch(Va, Vb, Vc, th0, om0)[:4]
    u = om + sim.physics.Kp * Vq            # the PI output the limiter clamps
    return Va, Vb, Vc, th, om, u


def per_window(model, ck, V, th_t, om_t, u, limit):
    """One (window RMS, saturated?) pair per RUN, kept run-wise on purpose.

    Pooling windows immediately would hide the decomposition: a window can be clean and
    still be wrong because an EARLIER window in the same run saturated and the recurrent
    handover carried the error forward. Runs that never saturate are the controlled
    comparison -- there famN's truth is bit-identical to famR's."""
    W, S = ck["data_meta"]["W"], ck["data_meta"]["S"]
    out = []
    for r in range(th_t.shape[0]):
        pth, _ = rollout(model, ck, (V[0][r], V[1][r], V[2][r]),
                         th_t[r, 0].float(), om_t[r, 0].float())
        e = (pth.double() - th_t[r]).reshape(W, S).pow(2).mean(1).sqrt().numpy()
        sat = (np.zeros(W, bool) if limit is None
               else (u[r].reshape(W, S).abs().max(1).values > limit).numpy())
        out.append((e, sat))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="24_limiter_cost.png")
    a = p.parse_args()

    runs = {}
    for fam, limit in (("famR_W40", None), ("famN_W40", LIMIT)):
        Va, Vb, Vc, th_t, om_t, u = truth(limit)
        paths = sorted(q for q in glob.glob(f"runs/{fam}_*sp0.pth") if "_L" not in q)
        if not paths:
            raise SystemExit(f"no plain checkpoints for {fam}")
        acc = []
        for q in paths:
            m, ck = load_f32(q)
            acc += per_window(m, ck, (Va, Vb, Vc), th_t, om_t, u, limit)
            torch.set_default_dtype(torch.float64)
        runs[fam] = acc
        print(f"{fam}: {len(paths)} seeds x {len(acc)//len(paths)} runs")
    PS.pll_constants.freq_limit = None

    R, Nn = runs["famR_W40"], runs["famN_W40"]
    quiet = [i for i, (e, sat) in enumerate(Nn) if not sat.any()]
    noisy = [i for i, (e, sat) in enumerate(Nn) if sat.any()]
    first = [int(np.argmax(sat)) for e, sat in Nn if sat.any()]

    groups = [
        ("famR\nno limiter", np.concatenate([e for e, _ in R]), "tab:blue"),
        ("famN\nruns that NEVER\nsaturate", np.concatenate([Nn[i][0] for i in quiet]), "tab:green"),
        ("famN\nclean windows,\nruns that DO", np.concatenate([Nn[i][0][~Nn[i][1]] for i in noisy]), "tab:orange"),
        ("famN\nSATURATED\nwindows", np.concatenate([Nn[i][0][Nn[i][1]] for i in noisy]), "tab:red"),
    ]
    ref = np.median(groups[0][1])
    print(f"\nruns that never saturate: {len(quiet)}/{len(Nn)}  |  first saturated window: "
          f"median {int(np.median(first))} of 40")
    print(f"\n{'group':44s} {'n':>7s} {'median':>11s} {'p90':>11s} {'vs famR':>9s}")
    for lab, v, _ in groups:
        print(f"{lab.replace(chr(92)+chr(110), ' '):44s} {len(v):7d} {np.median(v):11.3e} "
              f"{np.quantile(v, .9):11.3e} {np.median(v)/ref:8.2f}x")

    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
    for i, (lab, v, c) in enumerate(groups):
        ax[0].scatter(np.full(len(v), i) + rng.uniform(-.17, .17, len(v)), v,
                      s=3, alpha=.10, color=c, zorder=2)
        ax[0].hlines(np.median(v), i - .32, i + .32, color=c, lw=3.2, zorder=4)
        ax[0].vlines(i, np.quantile(v, .1), np.quantile(v, .9), color=c, lw=1.7, zorder=3)
        ax[0].annotate(f"{np.median(v):.2e}\n{np.median(v)/ref:.2f}x",
                       (i, np.median(v)), textcoords="offset points", xytext=(17, -2),
                       fontsize=9, color=c, fontweight="bold")
    ax[0].set_xticks(range(4))
    ax[0].set_xticklabels([g[0].replace(chr(92)+chr(110), chr(10)) for g in groups], fontsize=8.5)
    ax[0].set_yscale("log"); ax[0].grid(alpha=.3, which="both")
    ax[0].set_ylabel("per-window $\\theta$ RMS [rad]")
    ax[0].set_title("bar = median, whisker = p10-p90, 4 seeds x 12 runs x 40 windows",
                    fontsize=10)

    lo = min(g[1].min() for g in groups); hi = max(g[1].max() for g in groups)
    bins = np.logspace(np.log10(lo), np.log10(hi), 60)
    for lab, v, c in groups:
        ax[1].hist(v, bins=bins, histtype="step", lw=1.8, color=c, density=True,
                   label=f"{lab.replace(chr(92)+chr(110), ' ')}  (n={len(v)})")
    ax[1].set_xscale("log"); ax[1].set_xlabel("per-window $\\theta$ RMS [rad]")
    ax[1].set_ylabel("density"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, which="both")
    ax[1].set_title("Saturated windows are a separate population", fontsize=10)

    fig.suptitle("The limiter costs 2.1x EVERYWHERE and 22x where it fires.  "
                 f"Only {100*np.mean([s.mean() for _, s in Nn]):.1f}% of windows saturate, "
                 "so the aggregate hides both.", fontsize=12)
    fig.tight_layout()
    out = _graphs(a.out); fig.savefig(out, dpi=140)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
