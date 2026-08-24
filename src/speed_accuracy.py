"""Cost against accuracy for the four models in the deliverable set. graphs/23.

    python src/speed_accuracy.py            # ~5 min

The two levers the integrating team actually has are on different axes and 09_accuracy_vs_step.png shows
neither: it compares ONE fixed-gain W=40 model against the solvers. This puts all four
deliverable models on one plot so both trades are visible at once.

    W = 40 vs W = 20        half the network calls per simulated second
    fixed vs Kp,Ki INPUTS   retune the PLL without a retrain

EVERY MODEL IS SCORED THE SAME WAY, which is the only reason the numbers can be put on
one axis: 12 fresh trajectories, omega_0 in +/-2 rad/s (the warm co-simulation regime,
not cold acquisition), Kp=25 Ki=300, full 0.5 s recurrent rollout with the model's own
state handed forward and no ground truth anywhere in the loop. Per-family validation
numbers would NOT be comparable here -- that is F59/F61, and it is why this script
regenerates its own truth instead of reading the sweep records.

Cost is wall clock on THIS machine through `predict_window`, i.e. the deployed path with
autograd on, so it is directly comparable to the ms/sim-s already quoted in
graphs/Tunable_Kp_Ki_tests/README.md. It is a batch-1 number: the surrogate's advantage narrows as the
solver's vectorisation kicks in (53x at batch 1, 2.4x at batch 512 -- F41), so read this
as "one PLL inside one EMT loop", which is the co-simulation case.

Bands are min-max over every seed on disk. The seed spread on the deployed metric reaches
1.6x (F24), so a single checkpoint would carry more uncertainty than the W=40 vs W=20
difference this plot exists to show.
"""
import argparse
import glob
import time

import numpy as np
import torch

from PLL_Simulator import PLLSimulator
from common_test import load_f32, truth, rollout_rms
from paths import graphs as _graphs

DT, N, HORIZON = 100e-6, 5000, 0.5
THEIRS = (25.0, 300.0)

ARMS = [  # label, glob, colour, marker
    ("fixed gains, W=40",  "runs/famD_W40_*sp0.pth",   "tab:blue",   "o"),
    ("fixed gains, W=20",  "runs/famI_W20_*sp0.pth",   "tab:cyan",   "s"),
    ("Kp,Ki inputs, W=40", "runs/famJ_W40_*sp0_g.pth", "tab:red",    "o"),
    ("Kp,Ki inputs, W=20", "runs/famJ_W20_*sp0_g.pth", "tab:orange", "s"),
]


def solver_cost(n_runs=12, repeats=2):
    """The thing we are replacing, timed the same way: batch 1, same horizon, same dt."""
    torch.set_default_dtype(torch.float64)
    sim = PLLSimulator(dt=DT)
    sim.N, sim.n_runs = N, 1
    sim.t = (torch.arange(N) * DT).reshape(1, N)
    z = torch.zeros(1, 1)
    Va, Vb, Vc = sim._grid_phases(z, z, z)
    th0, om0 = torch.zeros(1), torch.zeros(1)
    with torch.no_grad():
        sim.simulate_batch(Va, Vb, Vc, th0, om0, scheme="trapezoid")      # warm up
        t0 = time.perf_counter()
        for _ in range(repeats):
            sim.simulate_batch(Va, Vb, Vc, th0, om0, scheme="trapezoid")
    return (time.perf_counter() - t0) / repeats / HORIZON * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_runs", type=int, default=12)
    p.add_argument("--out", default="23_speed_vs_accuracy.png")
    a = p.parse_args()

    Va, Vb, Vc, th_t, om_t = truth(*THEIRS, a.n_runs)
    sol_ms = solver_cost()
    print(f"trapezoid solver @100us, batch 1: {sol_ms:.1f} ms/simulated-second\n")

    out = {}
    for lab, pat, col, mk in ARMS:
        paths = sorted(glob.glob(pat))
        if not paths:
            raise SystemExit(f"no checkpoints matching {pat}")
        vals = [rollout_rms(*load_f32(q), (Va, Vb, Vc), th_t, om_t,
                            *THEIRS, timed=True) for q in paths]
        e = np.array([v[0] for v in vals]); m = np.array([v[1] for v in vals])
        out[lab] = (e, m, col, mk)
        print(f"{lab:20s} n={len(paths)}  theta {np.median(e):.3e} "
              f"[{e.min():.3e}, {e.max():.3e}]   {np.median(m):6.1f} ms/sim-s  "
              f"{sol_ms/np.median(m):5.1f}x faster than the solver")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    for lab, (e, m, col, mk) in out.items():
        ax.plot([m.min(), m.max()], [np.median(e)] * 2, color=col, lw=1.2, alpha=.6)
        ax.plot([np.median(m)] * 2, [e.min(), e.max()], color=col, lw=1.2, alpha=.6)
        ax.scatter(m, e, s=26, color=col, marker=mk, alpha=.45, zorder=3)
        ax.scatter([np.median(m)], [np.median(e)], s=190, color=col, marker=mk,
                   edgecolor="k", lw=1.3, zorder=4, label=f"{lab}  (n={e.size})")
        ax.annotate(f"{np.median(e):.2e} rad\n{np.median(m):.1f} ms/sim-s",
                    (np.median(m), np.median(e)), textcoords="offset points",
                    xytext=(12, -16), fontsize=8.5, color=col)
    ax.set_xscale("log"); ax.set_yscale("log")
    # The solver sits at ~989 ms/sim-s. Drawing it as a gridline would squash all four
    # models into 2% of the axis and hide the 2x spread this plot exists to show, so it
    # is an off-scale annotation instead.
    lo = min(m.min() for _, (e, m, c, k) in out.items())
    hi = max(m.max() for _, (e, m, c, k) in out.items())
    ax.set_xlim(lo * .82, hi * 1.55)
    ax.annotate(f"trapezoid solver, batch 1: {sol_ms:.0f} ms/sim-s\n"
                f"off-scale right — every model here is {sol_ms/hi:.0f}-{sol_ms/lo:.0f}x faster",
                xy=(.015, .02), xycoords="axes fraction", ha="left", va="bottom",
                fontsize=8.5, bbox=dict(fc="#eee", ec="k", lw=.6, pad=4))

    # The two trades, drawn as the arrows a reader would otherwise have to compute
    def med(lab): return np.median(out[lab][1]), np.median(out[lab][0])
    for a_lab, b_lab, txt, dy in (
            ("fixed gains, W=40", "Kp,Ki inputs, W=40",
             "tunable Kp,Ki\n{:.1f}x error, +{:.0f}% cost", 0),
            ("fixed gains, W=20", "Kp,Ki inputs, W=20",
             "tunable Kp,Ki\n{:.1f}x error, +{:.0f}% cost", 0)):
        (x0, y0), (x1, y1) = med(a_lab), med(b_lab)
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="grey", lw=1.4, ls=":"))
        ax.text(max(x0, x1) * 1.05, (y0 * y1) ** .5,
                txt.format(y1 / y0, (x1 / x0 - 1) * 100), fontsize=8, color="dimgrey",
                ha="left", va="center")
    (x0, y0), (x1, y1) = med("fixed gains, W=40"), med("fixed gains, W=20")
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color="grey", lw=1.4, ls=":"))
    ax.text((x0 * x1) ** .5, (y0 * y1) ** .5 * 1.06,
            f"W=40 $\\rightarrow$ W=20\n{x0/x1:.2f}x faster, {y1/y0:.2f}x error",
            fontsize=8, color="dimgrey", ha="center", va="bottom")
    ax.set_xlabel("cost [ms per simulated second, batch 1, deployed path]")
    ax.set_ylabel("deployed $\\theta$ RMS over 0.5 s [rad]")
    ax.set_title("Both trades on one axis: window length buys speed, tunable gains cost "
                 "accuracy\n"
                 f"{a.n_runs} trajectories, $\\omega_0 \\in \\pm2$ rad/s, Kp=25 Ki=300, "
                 "full recurrent rollout; bars are min-max over seeds", fontsize=10.5)
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=9, loc="upper right", framealpha=.95)
    fig.tight_layout()
    path = _graphs(a.out)
    fig.savefig(path, dpi=140)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
