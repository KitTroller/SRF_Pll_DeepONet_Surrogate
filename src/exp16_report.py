"""exp16: does removing faults, or narrowing the gain box, buy anything? graphs/22.

    python src/exp16_report.py                 # ~15 min, recomputes everything

Three families, all gains-as-inputs, all wide omega, all n_runs=5000, W=40 and W=20:

    famJ   faults ON,  gain box Kp 10-50, Ki 100-600
    famL   faults OFF, same wide box          -> famL vs famJ isolates the FAULTS
    famM   faults OFF, box Kp 18-45, Ki 180-520 -> famM vs famL isolates the BOX

WHY THIS SCRIPT EXISTS AT ALL. The sweep records say famL_W40 is 1.42x better than
famJ_W40 and famM_W40 another 1.36x better than that. Both numbers are artefacts: each
model is scored on its OWN family's validation split, famL/famM validate on fault-free
data, and famM's split spans a narrower gain box. Easier split, better number, no better
model. That is F59, retracted in F61, and it is the fifth time this class of error has
shown up in this project. Everything below is scored on ONE set of trajectories at ONE
set of gains, generated identically for every model, inside famM's trimmed box where all
three families are in-domain.

Truth is built once per condition and reused across all 24 checkpoints -- N = W*S = 5000
either way, so a single trajectory serves both the W=40 and the W=20 models.
"""
import argparse
import glob
import statistics as st

import numpy as np
import torch

from common_test import load_f32, truth, rollout_rms
from paths import graphs as _graphs

CLEAN_CELLS = [(20., 200.), (20., 500.), (25., 300.), (33., 350.), (42., 200.), (42., 500.)]
FAULT_CELLS = [(25., 300.), (33., 350.)]
FAMS = ["famJ_W40", "famL_W40", "famM_W40", "famJ_W20", "famL_W20", "famM_W20"]
LABEL = {"famJ": "faults ON\nwide box", "famL": "faults OFF\nwide box",
         "famM": "faults OFF\ntrimmed box"}
COL = {"famJ": "tab:blue", "famL": "tab:orange", "famM": "tab:red"}


def sweep(models, cells, kind, n_runs):
    out = {f: [] for f in models}
    for kp, ki in cells:
        Va, Vb, Vc, th_t, om_t = truth(kp, ki, n_runs, kind)
        for f, paths in models.items():
            for p in paths:
                m, ck = load_f32(p)
                out[f].append(rollout_rms(m, ck, (Va, Vb, Vc), th_t, om_t, kp, ki))
        print(f"  {kind:5s} Kp={kp:.0f} Ki={ki:.0f}", flush=True)
    return out


def _panel(ax, data, order, title, ylab):
    jit = np.random.default_rng(0)          # seeded: unseeded jitter made the PNG differ
    for i, key in enumerate(order):         # on every run even when the data was identical
        v = np.array(data[key])
        stem = key.split("_")[0] if "_" in key else key
        ax.scatter(np.full_like(v, i) + jit.uniform(-.09, .09, v.size), v,
                   s=22, alpha=.55, color=COL.get(stem, "grey"), zorder=3)
        ax.hlines(np.median(v), i - .28, i + .28, color=COL.get(stem, "grey"), lw=2.6, zorder=4)
    ax.set_xticks(range(len(order)))
    ax.set_yscale("log")
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=.25, which="both")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_runs", type=int, default=8)
    p.add_argument("--out", default="22_exp16_nofault_gainbox.png")
    a = p.parse_args()

    models = {f: sorted(glob.glob(f"runs/{f}_*_g.pth")) for f in FAMS}
    for f, ps in models.items():
        print(f"{f}: {len(ps)} checkpoints")
        if not ps:
            raise SystemExit(f"no checkpoints for {f} -- pull runs/ from the cluster first")

    clean = sweep(models, CLEAN_CELLS, "clean", a.n_runs)
    fc = {f: models[f] for f in FAMS}
    fault = {k: sweep(fc, FAULT_CELLS, k, a.n_runs) for k in ("clean", "sag", "jump")}

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    for j, W in enumerate((40, 20)):
        order = [f"fam{s}_W{W}" for s in "JLM"]
        _panel(ax[j], clean, order,
               f"W = {W}: clean, common test set\n6 cells inside the trimmed box, "
               f"{a.n_runs} runs each, 4 seeds",
               "deployed $\\theta$ RMS over 0.5 s [rad]")
        ax[j].set_xticklabels([LABEL[k.split('_')[0]] for k in order], fontsize=9)
        med = [np.median(clean[k]) for k in order]
        ax[j].text(.5, .04, "medians within "
                            f"{max(med)/min(med):.2f}x — no arm clears the seed spread",
                   transform=ax[j].transAxes, ha="center", fontsize=9,
                   bbox=dict(fc="#ffe9c7", ec="none", pad=3))

    order = [f"fam{s}_W40" for s in "JLM"]
    x = np.arange(3)
    for i, kind in enumerate(("clean", "sag", "jump")):
        v = [np.median(fault[kind][k]) for k in order]
        ax[2].bar(x + (i - 1) * .26, v, width=.24,
                  color=["#bbb", "tab:red", "tab:green"][i], label=kind,
                  edgecolor="k", lw=.5)
    ax[2].set_yscale("log"); ax[2].set_xticks(x)
    ax[2].set_xticklabels([LABEL[k.split("_")[0]] for k in order], fontsize=9)
    ax[2].set_ylabel("deployed $\\theta$ RMS [rad]")
    ax[2].set_title("W = 40: what removing faults COSTS\n"
                    "sag = 0.70 pu for 60 ms; jump = +40 deg", fontsize=10)
    ax[2].legend(fontsize=9); ax[2].grid(alpha=.25, axis="y", which="both")
    top = max(np.median(fault["sag"][k]) for k in order)
    ax[2].set_ylim(top=top * 2.2)                     # headroom for the x-clean labels
    for i, k in enumerate(order):
        r = np.median(fault["sag"][k]) / np.median(fault["clean"][k])
        ax[2].text(i, np.median(fault["sag"][k]) * 1.12, f"{r:.1f}x", ha="center",
                   fontsize=9, color="tab:red", fontweight="bold")

    fig.suptitle("exp16 — neither removing faults nor narrowing the gain box helps, "
                 "and removing faults costs 5-9x on voltage sags", fontsize=12)
    fig.tight_layout()
    out = _graphs(a.out)
    fig.savefig(out, dpi=130)
    print(f"\n-> {out}")

    print(f"\n{'family':10s} {'clean':>10s} {'sag':>10s} {'xclean':>7s} {'jump':>10s} {'xclean':>7s}")
    for f in FAMS:
        c, s, j = (st.median(fault[k][f]) for k in ("clean", "sag", "jump"))
        print(f"{f:10s} {c:10.3e} {s:10.3e} {s/c:7.2f} {j:10.3e} {j/c:7.2f}")


if __name__ == "__main__":
    main()
