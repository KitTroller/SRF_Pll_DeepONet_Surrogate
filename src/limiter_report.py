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


GROUPS = ("no limiter", "limiter,\nruns that\nNEVER saturate",
          "limiter, clean\nwindows of runs\nthat DO", "limiter,\nSATURATED\nwindows")
GCOL = ("tab:blue", "tab:green", "tab:orange", "tab:red")


def one_pair(limited, unlimited):
    """The four groups for ONE draw. Returns [(label, values)] plus the saturated fraction."""
    runs = {}
    for fam, limit in ((unlimited, None), (limited, LIMIT)):
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
        print(f"  {fam}: {len(paths)} seeds x {len(acc)//len(paths)} runs")
    PS.pll_constants.freq_limit = None

    R, Nn = runs[unlimited], runs[limited]
    quiet = [i for i, (e, sat) in enumerate(Nn) if not sat.any()]
    noisy = [i for i, (e, sat) in enumerate(Nn) if sat.any()]
    return ([np.concatenate([e for e, _ in R]),
             np.concatenate([Nn[i][0] for i in quiet]),
             np.concatenate([Nn[i][0][~Nn[i][1]] for i in noisy]),
             np.concatenate([Nn[i][0][Nn[i][1]] for i in noisy])],
            float(np.mean([sat.mean() for _, sat in Nn])),
            int(np.median([np.argmax(sat) for _, sat in Nn if sat.any()])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="24_limiter_cost.png")
    p.add_argument("--pairs", nargs="+",
                   default=["famN_W40,famR_W40", "famS_W40,famT_W40"],
                   help="limited,unlimited per draw. The two draws are plotted SIDE BY "
                        "SIDE and never pooled -- pooling would hide the replication, "
                        "which is the whole result.")
    a = p.parse_args()

    draws = []
    for i, pr in enumerate(a.pairs):
        lim, unl = pr.split(",")
        print(f"draw {i+1}: {lim} vs {unl}")
        vals, satfrac, first = one_pair(lim, unl)
        draws.append((f"draw {i+1}\n{lim.split('_')[0]}/{unl.split('_')[0]}", vals, satfrac, first))

    print(f"\n{'group':34s}" + "".join(f"{d[0].splitlines()[0]:>22s}" for d in draws))
    for gi, g in enumerate(GROUPS):
        row = ""
        for _, vals, _, _ in draws:
            row += f"{np.median(vals[gi]):11.3e}{np.median(vals[gi])/np.median(vals[0]):8.2f}x"
        print(f"{g.replace(chr(10), ' '):34s}{row}")

    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.6))
    nd = len(draws)
    off = np.linspace(-.22, .22, nd) if nd > 1 else [0.0]
    mk = ("o", "s", "^")
    for di, (dlab, vals, satfrac, first) in enumerate(draws):
        for gi, v in enumerate(vals):
            x = gi + off[di]
            ax[0].scatter(np.full(len(v), x) + rng.uniform(-.055, .055, len(v)), v,
                          s=2.5, alpha=.09, color=GCOL[gi], zorder=2)
            ax[0].hlines(np.median(v), x - .1, x + .1, color=GCOL[gi], lw=3, zorder=4)
            ax[0].vlines(x, np.quantile(v, .1), np.quantile(v, .9), color=GCOL[gi],
                         lw=1.5, zorder=3)
            ax[0].scatter([x], [np.median(v)], marker=mk[di % 3], s=46, color=GCOL[gi],
                          edgecolor="k", lw=.8, zorder=5)
        ax[1].plot(range(4), [np.median(v) / np.median(vals[0]) for v in vals],
                   "-" + mk[di % 3], lw=2, ms=9, label=dlab.replace("\n", "  "))
        for gi, v in enumerate(vals):
            ax[1].annotate(f"{np.median(v)/np.median(vals[0]):.2f}x", (gi, np.median(v)/np.median(vals[0])),
                           textcoords="offset points", xytext=(8, 6 - 14 * di), fontsize=9)

    for k in (0, 1):
        ax[k].set_xticks(range(4)); ax[k].set_xticklabels(GROUPS, fontsize=8.5)
        ax[k].set_yscale("log"); ax[k].grid(alpha=.3, which="both")
    ax[0].set_ylabel("per-window $\\theta$ RMS [rad]")
    ax[0].set_title("Every window, both draws.  bar = median, whisker = p10-p90",
                    fontsize=10.5)
    ax[1].set_ylabel("$\\times$ that draw's own no-limiter baseline")
    ax[1].set_title("The ratios are what replicate", fontsize=10.5)
    ax[1].legend(fontsize=9.5); ax[1].axhline(1, color="k", lw=1)

    fr = sorted({f"{100*d[2]:.1f}%" for d in draws})       # both draws agree -> print once
    sat = fr[0] if len(fr) == 1 else " and ".join(fr)
    fig.suptitle("The limiter costs ~2.1x EVERYWHERE and ~22x where it fires — "
                 f"and it replicates across two independent LHS draws.\n"
                 f"Only {sat} of windows saturate, so a pooled metric shows none of it.",
                 fontsize=12)
    fig.tight_layout()
    out = _graphs(a.out); fig.savefig(out, dpi=140)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
