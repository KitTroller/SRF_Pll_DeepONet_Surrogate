"""exp32 on the same axis as exp29-31: data (5k / 10k / 20k) x anchor (none / after / trained in).

    python src/analysis/exp32_report.py            # both common tests, graphs/30 (~4 min)

LEFT: famO's 150 val runs. RIGHT: famQ's 150 val runs. Each split is also one family's
early-stopping set -- famO's favours the 5k models, famQ's the 10k ones -- so a result is
only trusted when it holds on BOTH. Each panel is normalised to its own 5k baseline median.

Scores whatever exp32 checkpoints exist, so it can be run while the array is still going --
every group carries its seed count and the figure prints it. Reference groups (baseline,
famQ, and both with the anchor added afterwards) are read from round29_31's saved JSON
rather than recomputed; they were measured on the same runs with the same code.

Colour = how much data. Shape = what was done about the anchor. The comparison that
matters, and the one exp30 got wrong at 5k, is diamond vs square at the same colour:
anchor added at inference against anchor trained in.
"""

# src/ on the path: these scripts live in src/analysis/ but import the pipeline
# modules (paths, PLL_Simulator, train_pll, sweep) that stay in src/. Running
# `python src/analysis/foo.py` puts src/analysis on sys.path, not src/.
# Same pattern as hpc/generate_family.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import argparse
import json
import statistics as st

import numpy as np

import round29_31 as R
from paths import ROOT, graphs as _graphs

GREY, ORANGE, AQUA, BLUE = "#555555", "#eb6834", "#1baf7a", "#2a78d6"
#        key                  label                     runs     colour  marker
COLS = [("base_cpu",          "5k\nplain",              "5k",    GREY,   "o"),
        ("base_cpu+posthoc",  "5k\nanchor after",       "5k",    GREY,   "D"),
        ("anchor",            "5k\nanchor trained in",  "5k",    GREY,   "s"),
        ("famQ",              "10k\nplain",             "10k",   ORANGE, "o"),
        ("famQ+posthoc",      "10k\nanchor after",      "10k",   ORANGE, "D"),
        ("famQ_ao",           "10k\nanchor trained in",  "10k",  ORANGE, "s"),
        ("famO20k",           "20k\nplain",             "20k",   AQUA,   "o"),
        ("famO20k+posthoc",   "20k\nanchor after",      "20k",   AQUA,   "D"),
        ("famO20k_ao",        "20k\nanchor trained in", "20k",   AQUA,   "s"),
        ("famO40k",           "40k\nplain",             "40k",   BLUE,   "o"),   # exp33
        ("famO40k+posthoc",   "40k\nanchor after",      "40k",   BLUE,   "D"),
        ("famO40k_ao",        "40k\nanchor trained in", "40k",   BLUE,   "s")]


def score(split, n_runs):
    """Reference groups from round29_31's saved JSON + every exp32/33 checkpoint that exists.
    A group already in this split's saved exp32 JSON with the same seed count is reused, so
    only newly landed seeds are scored (a full rescore is ~4 min per split)."""
    saved = ROOT / "Hyperparameter_sweep" / ("round29_31.json" if split == "famO_W40.npz"
                                             else "round29_31_split_famQ_W40.json")
    ref = json.load(open(saved))["part1"]
    out = ROOT / "Hyperparameter_sweep" / f"exp32_{_Path(split).stem}.json"
    prev = json.load(open(out)) if out.exists() else {}

    # only the exp32/33 checkpoints that exist -- the arrays may still be running
    have = lambda n: (ROOT / "runs" / n).exists()
    groups = {g: [n for n in names if have(n)] for g, names in {
        "famO20k":    [f"famO20k{R.T.format(n=20000, s=s, x='')}" for s in range(8)],
        "famO20k_ao": [f"famO20k{R.T.format(n=20000, s=s, x='_ao')}" for s in range(8)],
        "famQ_ao":    [f"famQ{R.T.format(n=10000, s=s, x='_ao')}" for s in range(8)],
        "famO40k":    [f"famO40k{R.T.format(n=40000, s=s, x='')}" for s in range(8)],
        "famO40k_ao": [f"famO40k{R.T.format(n=40000, s=s, x='_ao')}" for s in range(8)],
    }.items()}
    groups = {g: v for g, v in groups.items() if v}
    R.GROUPS = {g: v for g, v in groups.items() if len(prev.get(g, [])) != len(v)}
    for g in groups.keys() - R.GROUPS.keys():
        ref.update({k: prev[k] for k in (g, g + "+posthoc") if k in prev})
    print(f"{split}: reusing " + ", ".join(sorted(groups.keys() - R.GROUPS.keys())) + " | scoring "
          + ", ".join(f"{g} ({len(v)} seeds)" for g, v in R.GROUPS.items()))
    if R.GROUPS:
        new, _ = R.part1(n_runs, split)
        ref.update(new)
    json.dump(ref, open(out, "w"), indent=1)
    return ref


def panel(ax, ref, split, n_runs):
    base = st.median(r["rms"] for r in ref["base_cpu"])
    print(f"\ncommon test: {split}'s {n_runs} val runs   (x = baseline median / group median)")
    for k, lbl, *_ in COLS:
        if k not in ref:
            print(f"  {lbl.replace(chr(10), ' '):26s} -- not trained yet"); continue
        v = [r["rms"] for r in ref[k]]; mx = [r["max"] for r in ref[k]]
        print(f"  {lbl.replace(chr(10), ' '):26s} n={len(v)}  {st.median(v):.3e} "
              f"[{min(v):.2e}-{max(v):.2e}]  {base / st.median(v):5.2f}x   worst {st.median(mx):.4f}")

    rng = np.random.default_rng(0)
    lo, hi = min(r["rms"] for r in ref["base_cpu"]), max(r["rms"] for r in ref["base_cpu"])
    ax.axhspan(lo, hi, color=GREY, alpha=.08, lw=0)
    ticks, labels = [], []
    for i, (k, lbl, _, c, m) in enumerate(COLS):
        ticks.append(i); labels.append(lbl + (f"\nn={len(ref[k])}" if k in ref else "\nrunning"))
        if k not in ref:
            ax.annotate("still\nrunning", (i, base * 0.62), ha="center", fontsize=8.5, color=GREY)
            continue
        v = [r["rms"] for r in ref[k]]
        ax.scatter(i + rng.uniform(-.13, .13, len(v)), v, s=52, color=c, marker=m,
                   edgecolor="white", linewidth=1.1, zorder=3)
        med = st.median(v)
        ax.hlines(med, i - .3, i + .3, color="#222222", lw=2, zorder=4)
        ax.annotate(f"{base / med:.2f}x", (i, max(v)), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=9.5, color="#222222")
    ax.set_yscale("log"); ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=8.6)
    ax.set_xlim(-.6, len(COLS) - .4)       # keep "still running" columns on the axis
    ax.set_ylabel(f"deployed theta RMS on {_Path(split).stem}'s {n_runs} val runs [rad]", fontsize=9.5)
    ax.grid(alpha=.25, axis="y", which="both")
    ax.spines[["top", "right"]].set_visible(False)
    for h, lab in [("o", "plain"), ("D", "anchor added at inference"), ("s", "anchor trained in")]:
        ax.scatter([], [], marker=h, s=52, color="#888888", edgecolor="white", label=lab)
    ax.legend(fontsize=8.5, loc="lower left", title="marker", title_fontsize=8)
    ranked = sorted((st.median(r["rms"] for r in ref[k]), lbl) for k, lbl, *_ in COLS if k in ref)
    (b1, l1), (b2, l2) = ranked[0], ranked[1]
    name = lambda l: l.replace(chr(10), " ")
    # within 2% is inside the seed spread -- call it a tie rather than crown a winner
    best = (f"{name(l1)} ({base / b1:.2f}x) and {name(l2)} ({base / b2:.2f}x) tie" if b2 / b1 < 1.02
            else f"best: {name(l1)} at {base / b1:.2f}x")
    ax.set_title(f"test runs from {_Path(split).stem} -- {best}", fontsize=10)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", nargs="+", default=["famO_W40.npz", "famQ_W40.npz"])
    p.add_argument("--n_runs", type=int, default=150)
    p.add_argument("--out", default="30_exp32.png")
    p.add_argument("--plot_only", action="store_true",
                   help="redraw from the saved Hyperparameter_sweep/exp32_<split>.json, no rescoring")
    a = p.parse_args()

    load = lambda sp: json.load(open(ROOT / "Hyperparameter_sweep" / f"exp32_{_Path(sp).stem}.json"))
    refs = [(sp, load(sp) if a.plot_only else score(sp, a.n_runs)) for sp in a.splits]
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(refs), figsize=(13.5 * len(refs), 6.2), squeeze=False)
    for ax, (sp, ref) in zip(axes[0], refs):
        panel(ax, ref, sp, a.n_runs)
    fig.suptitle("exp32-33: without the anchor in training, data flattens after 20k (20k -> 40k: 1.01-1.06x); "
                 "WITH it, 20k -> 40k still pays 1.17-1.23x. Training with the anchor loses at 5-10k, "
                 "ties at 20k, and wins at 40k\n"
                 "colour = training runs, marker = anchor treatment; bar = median over seeds; number = "
                 "that panel's 5k-plain median / group median (>1 is better)", fontsize=11)
    fig.tight_layout()
    out = _graphs(_Path(a.out)); out.parent.mkdir(exist_ok=True); fig.savefig(out, dpi=150)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
