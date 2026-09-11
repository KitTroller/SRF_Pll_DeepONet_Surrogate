"""Does the network have enough capacity? graphs/26.  exp17's grid.

    python src/capacity_grid.py

depth {2,3,4} x interior width {32,64,128}, 4 seeds, on two families that differ ONLY by
the frequency limiter:

    famN_W40   limited   -- the piecewise target
    famR_W40   unlimited -- the control, same --lhs_seed 21

Both are read from their OWN sweep records, which is legitimate here because every cell
inside a panel shares one dataset and one validation split: the only thing that varies is
the architecture. Comparing the two PANELS is a different matter -- those are different
physics, and the honest cross-family number is in graphs/24 (F65), not here.

WHY THIS WAS WORTH RUNNING AT ALL. F46 reported capacity as flat and the README repeated
it, but `hidden_dim` only ever moved `sizes[-1]` -- the latent contraction width. The
interior width and the depth had never been varied (F63). They are not flat.
"""

# src/ on the path: these scripts live in src/analysis/ but import the pipeline
# modules (paths, PLL_Simulator, train_pll, sweep) that stay in src/. Running
# `python src/analysis/foo.py` puts src/analysis on sys.path, not src/.
# Same pattern as hpc/generate_family.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import argparse
import glob
import json
import statistics as st
from collections import defaultdict

import numpy as np

from paths import graphs as _graphs
from sweep import load as _load

DEPTHS, WIDTHS = (2, 3, 4), (32, 64, 128)
COL = {32: "tab:orange", 64: "tab:blue", 128: "tab:green"}


def cells(results_dir):
    g = defaultdict(list)
    for r in _load(results_dir):
        if r.get("status") == "ok" and r.get("n_layers") and r.get("width"):
            g[(r["n_layers"], r["width"])].append(r)
    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="26_capacity_grid.png")
    p.add_argument("--metric", default="rollout_full_rms")
    p.add_argument("--min_seeds", type=int, default=4,
                   help="cells with fewer get their MEDIAN drawn as a hollow marker and "
                        "no shaded band. The band is the part that lies: a min/max range "
                        "over 2 draws is narrower than over 4 by construction, which is "
                        "how the hidden-dim figure misled once. The median of 2 tight "
                        "draws is worth seeing; the spread of 2 draws is not")
    p.add_argument("--hide_incomplete", action="store_true",
                   help="drop under-seeded cells entirely instead of marking them")
    a = p.parse_args()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(21, 5))
    fams = [("famN_W40_cap", "famN — limiter, FIXED gains"),
            ("famO_W40_cap", "famO — limiter + TUNABLE gains\n(the deliverable)"),
            ("famR_W40_cap", "famR — unlimited control")]
    store = {}

    for i, (d, title) in enumerate(fams):
        g = cells("sweeps_" + d)
        base = g.get((2, 64))
        b = st.median(r[a.metric] for r in base)
        store[d] = {}
        for w in WIDTHS:
            xs, ys, lo, hi = [], [], [], []
            px, py, pn = [], [], []                 # under-seeded: median only, no band
            for L in DEPTHS:
                v = g.get((L, w), [])
                if not v or (a.hide_incomplete and len(v) < a.min_seeds):
                    continue
                q = sorted(r[a.metric] for r in v)
                store[d][(L, w)] = (float(np.median(q)) / b, len(v))
                if len(v) < a.min_seeds:
                    px.append(L); py.append(np.median(q)); pn.append(len(v))
                else:
                    xs.append(L); ys.append(np.median(q)); lo.append(q[0]); hi.append(q[-1])
            if not xs and not px:
                continue
            if xs:
                ax[i].plot(xs, ys, "-o", color=COL[w], lw=2, ms=7, label=f"width {w}")
                ax[i].fill_between(xs, lo, hi, color=COL[w], alpha=.15)
            for L, y, n in zip(px, py, pn):
                # hollow, dashed, and labelled with n. No band -- the median of a tight
                # pair is informative, its min/max is not a spread estimate.
                if xs:
                    ax[i].plot([xs[-1], L], [ys[-1], y], ls=":", color=COL[w], lw=1.6)
                ax[i].plot([L], [y], "o", mfc="none", mec=COL[w], mew=2, ms=9,
                           label=None if xs else f"width {w}")
                ax[i].annotate(f"n={n}", (L, y), textcoords="offset points",
                               xytext=(7, -11), fontsize=8, color=COL[w])
        ax[i].axhline(b, color="k", ls=":", lw=1.2)
        ax[i].text(2.02, b, " default (L2, w64)", va="bottom", fontsize=8.5)
        ax[i].set_yscale("log"); ax[i].set_xticks(DEPTHS)
        ax[i].set_xlabel("interior depth (`--n_layers`)")
        ax[i].set_ylabel("deployed $\\theta$ RMS over 0.5 s [rad]")
        ax[i].set_title(title, fontsize=10.5)
        ax[i].grid(alpha=.3, which="both"); ax[i].legend(fontsize=9)

    ys = [y for x in ax[:3] for l in x.get_lines() for y in np.asarray(l.get_ydata())
          if np.isfinite(y)]
    lo, hi = min(ys), max(ys)
    for i in (0, 1, 2):
        ax[i].set_ylim(lo * .8, hi * 1.25)          # same axis or the panels do not compare

    # third panel: the gain relative to each family's OWN default, which is the thing that
    # is actually comparable between two different physics
    keys = [k for k in store["famN_W40_cap"] if k in store["famR_W40_cap"]]
    keys.sort(key=lambda k: (k[1], k[0]))
    x = np.arange(len(keys)); w_ = .27
    for off, d, c, lab in ((-w_, "famN_W40_cap", "tab:red", "famN (limiter, fixed)"),
                           (0.0, "famO_W40_cap", "tab:purple", "famO (limiter + gains)"),
                           (w_, "famR_W40_cap", "tab:blue", "famR (unlimited)")):
        vals = [1 / store[d][k][0] if k in store[d] else 0 for k in keys]
        ns = [store[d][k][1] if k in store[d] else 0 for k in keys]
        # hatched = fewer seeds than min_seeds. The height is a median and stands; the
        # hatch says do not quote it as settled.
        ax[3].bar(x + off, vals, w_ * .92, color=c, label=lab, edgecolor="k", lw=.5,
                  hatch=None)
        for xi, v, n in zip(x + off, vals, ns):
            if 0 < n < a.min_seeds:
                ax[3].bar([xi], [v], w_ * .92, color="none", edgecolor="k", lw=.5,
                          hatch="///")
                ax[3].annotate(f"n={n}", (xi, v), textcoords="offset points",
                               xytext=(0, 3), ha="center", fontsize=7)
    ax[3].axhline(1, color="k", lw=1)
    ax[3].set_xticks(x); ax[3].set_xticklabels([f"L{L}\nw{w}" for L, w in keys], fontsize=8.5)
    ax[3].set_ylabel("improvement over that family's own default  ($\\times$)")
    ax[3].set_title("Both LIMITED families gain ~2x from depth.\nThe unlimited one does not.",
                    fontsize=10.5)
    ax[3].legend(fontsize=8.5); ax[3].grid(alpha=.3, axis="y")

    fig.suptitle("Interior depth and width were never tested — `hidden_dim` only moved the "
                 "latent dimension (F63).  They are not flat.", fontsize=12)
    fig.tight_layout()
    out = _graphs(a.out); fig.savefig(out, dpi=140)
    print(f"-> {out}")

    print(f"\n{'cell':10s} {'famN':>11s} {'famO':>11s} {'famR':>11s}   "
          f"(x that family's own default; ! = fewer than {a.min_seeds} seeds)")
    for k in keys:
        row = ""
        for d in ("famN_W40_cap", "famO_W40_cap", "famR_W40_cap"):
            if k not in store[d]:
                row += f"{'--':>11s}"
            else:
                r_, n_ = store[d][k]
                row += f"{r_:9.2f}x{'!' if n_ < a.min_seeds else ' '}"
        print(f"L{k[0]}_w{k[1]:<6d} {row}")


if __name__ == "__main__":
    main()
