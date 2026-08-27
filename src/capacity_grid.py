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
                   help="cells with fewer are DROPPED, not plotted with a short bar -- a "
                        "min/max range over 2 draws is narrower than over 4 by "
                        "construction, which is how the hidden-dim figure misled once")
    a = p.parse_args()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    fams = [("famN_W40_cap", "famN — LIMITED (piecewise target)"),
            ("famR_W40_cap", "famR — unlimited control")]
    store = {}

    for i, (d, title) in enumerate(fams):
        g = cells("sweeps_" + d)
        base = g.get((2, 64))
        b = st.median(r[a.metric] for r in base)
        store[d] = {}
        for w in WIDTHS:
            xs, ys, lo, hi = [], [], [], []
            for L in DEPTHS:
                v = g.get((L, w), [])
                if len(v) < a.min_seeds:
                    continue
                q = sorted(r[a.metric] for r in v)
                xs.append(L); ys.append(np.median(q)); lo.append(q[0]); hi.append(q[-1])
                store[d][(L, w)] = np.median(q) / b
            if not xs:
                continue
            ax[i].plot(xs, ys, "-o", color=COL[w], lw=2, ms=7, label=f"width {w}")
            ax[i].fill_between(xs, lo, hi, color=COL[w], alpha=.15)
        ax[i].axhline(b, color="k", ls=":", lw=1.2)
        ax[i].text(2.02, b, " default (L2, w64)", va="bottom", fontsize=8.5)
        ax[i].set_yscale("log"); ax[i].set_xticks(DEPTHS)
        ax[i].set_xlabel("interior depth (`--n_layers`)")
        ax[i].set_ylabel("deployed $\\theta$ RMS over 0.5 s [rad]")
        ax[i].set_title(title, fontsize=10.5)
        ax[i].grid(alpha=.3, which="both"); ax[i].legend(fontsize=9)

    ys = [y for x in ax[:2] for l in x.get_lines() for y in np.asarray(l.get_ydata())
          if np.isfinite(y)]
    lo, hi = min(ys), max(ys)
    for i in (0, 1):
        ax[i].set_ylim(lo * .8, hi * 1.25)          # same axis or the panels do not compare

    # third panel: the gain relative to each family's OWN default, which is the thing that
    # is actually comparable between two different physics
    keys = [k for k in store["famN_W40_cap"] if k in store["famR_W40_cap"]]
    keys.sort(key=lambda k: (k[1], k[0]))
    x = np.arange(len(keys))
    ax[2].bar(x - .2, [1 / store["famN_W40_cap"][k] for k in keys], .38,
              color="tab:red", label="famN (limited)", edgecolor="k", lw=.5)
    ax[2].bar(x + .2, [1 / store["famR_W40_cap"][k] for k in keys], .38,
              color="tab:blue", label="famR (control)", edgecolor="k", lw=.5)
    ax[2].axhline(1, color="k", lw=1)
    ax[2].set_xticks(x); ax[2].set_xticklabels([f"L{L}\nw{w}" for L, w in keys], fontsize=8.5)
    ax[2].set_ylabel("improvement over that family's own default  ($\\times$)")
    ax[2].set_title("Depth helps the LIMITED problem about twice as much", fontsize=10.5)
    ax[2].legend(fontsize=9); ax[2].grid(alpha=.3, axis="y")

    fig.suptitle("Interior depth and width were never tested — `hidden_dim` only moved the "
                 "latent dimension (F63).  They are not flat.", fontsize=12)
    fig.tight_layout()
    out = _graphs(a.out); fig.savefig(out, dpi=140)
    print(f"-> {out}")

    print(f"\n{'cell':10s} {'params':>8s} {'famN x base':>12s} {'famR x base':>12s} {'ratio':>8s}")
    for k in keys:
        n, r = store["famN_W40_cap"][k], store["famR_W40_cap"][k]
        print(f"L{k[0]}_w{k[1]:<6d} {'':>8s} {n:11.2f}x {r:11.2f}x {r/n:7.2f}x")


if __name__ == "__main__":
    main()
