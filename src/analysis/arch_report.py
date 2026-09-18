"""exp28: split trunk x gains placement on the deliverable. graphs/28.

    python src/analysis/arch_report.py

Four arms on famO_W40 at L3_w128, one dot per seed:

    baseline   shared trunk, Kp/Ki in the branch       (the deliverable)
    A          --split_trunk      one trunk basis per output
    B          --gains_on_trunk   Kp/Ki to the trunk   (Choi et al. Model 3)
    A+B        both

Every record comes from ONE directory, one dataset and one --split_seed, so the arms
share a validation split and compare directly -- no common-test re-scoring needed.

THREE PANELS, because they answer three different questions:
    deployed RMS   the headline -- the 40-window recurrent rollout
    worst case     a limiter claim is a peak claim (F71); depth once helped one and hurt
                   the other (F70), so both are always shown
    train_th       training loss at the best epoch. If an arm is worse HERE too it does not
                   fit its own training set, so the cost is not a generalisation gap that
                   more data or regularisation would close. Whether it CANNOT fit
                   (representation) or stops before it does (optimisation) is not
                   separable from these records: B also early-stops ~190 epochs sooner
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

import matplotlib.pyplot as plt
import numpy as np

from paths import graphs as _graphs
from sweep import load as _load

# baseline is the reference, so neutral ink; the three changes take categorical slots 1-3
ARMS = [("base", "baseline\n(deliverable)", "#555555"),
        ("A",    "A: split trunk",          "#2a78d6"),
        ("B",    "B: gains on trunk",       "#eb6834"),
        ("A+B",  "A+B: both",               "#1baf7a")]
PANELS = [("rollout_full_rms", "deployed θ RMS, 40-window rollout [rad]"),
          ("rollout_full_max", "worst case |θ error| [rad]"),
          ("train_th",         "training θ loss at best epoch")]


def arm(r):
    s, g = r.get("split_trunk", False), r.get("gains_on_trunk", False)
    return "A+B" if s and g else "A" if s else "B" if g else "base"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="sweeps_famO_W40_arch")
    p.add_argument("--out", default="28_architecture.png")
    a = p.parse_args()

    by = {k: [] for k, _, _ in ARMS}
    for r in _load(a.results_dir):
        by[arm(r)].append(r)
    n = {k: len(v) for k, v in by.items()}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    rng = np.random.default_rng(0)                    # fixed jitter: the figure is reproducible
    for ax, (m, ylabel) in zip(axes, PANELS):
        base = [r[m] for r in by["base"]]
        b_med = st.median(base)
        ax.axhspan(min(base), max(base), color="#555555", alpha=.08, lw=0, zorder=0)
        for i, (k, _, c) in enumerate(ARMS):
            v = [r[m] for r in by[k]]
            if not v:
                continue
            x = i + rng.uniform(-.14, .14, len(v))
            ax.scatter(x, v, s=46, color=c, edgecolor="white", linewidth=1.2, zorder=3)
            med = st.median(v)
            ax.hlines(med, i - .28, i + .28, color="#222222", lw=2, zorder=4)
            if k != "base":
                ax.annotate(f"{med / b_med:.2f}x", (i, max(v)), xytext=(0, 7),
                            textcoords="offset points", ha="center", fontsize=9, color="#222222")
        ax.set_yscale("log")
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([f"{lbl}\nn={n[k]}" for k, lbl, _ in ARMS], fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=.25, which="both", axis="y")
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(y=.12)
    axes[0].set_title("the headline", fontsize=10)
    axes[1].set_title("the peak -- a limiter claim is a peak claim", fontsize=10)
    axes[2].set_title("worse on its OWN training set too:\na fitting problem, not a generalisation gap",
                      fontsize=10)

    rms = {k: st.median(r["rollout_full_rms"] for r in v) for k, v in by.items() if v}
    fig.suptitle(
        "exp28, famO L3_w128: neither change helps. Both make the model worse -- "
        + ", ".join(f"{k} {rms[k] / rms['base']:.2f}x" for k in ("A", "B", "A+B") if k in rms)
        + " on deployed RMS\n"
        "One dot per seed, bar = median, number = median / baseline median (>1 is worse), "
        "grey band = baseline's seed range. Same dataset and split in every arm.",
        fontsize=10.5)
    fig.tight_layout()
    out = _graphs(_Path(a.out))
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"-> {out}")
    for k, _, _ in ARMS:
        if by[k]:
            print(f"{k:5s} n={n[k]}  " + "  ".join(
                f"{m} {st.median(r[m] for r in by[k]):.3e}" for m, _ in PANELS))


if __name__ == "__main__":
    main()
