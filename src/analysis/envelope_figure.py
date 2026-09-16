"""THE head-to-head figure: graphs/12. Subsumes the retired figure 09.

    python src/envelope_figure.py runs/famD_W40_n5000_W40_F4_mf503_wp0.3_s0sp0.pth

Two panels, both from ONE `head_to_head` call:

  LEFT   |theta error| against time, mean over runs. Answers "does the recurrent handover
         accumulate error, or does it saturate?" -- the question a whole-window operator
         has to answer and a one-step map never faces. This was figure 09's left panel,
         recomputed there without the paper's NN; it is strictly better here, so 09 is
         retired rather than regenerated.
  RIGHT  theta RMS against compute. 09's right panel showed the same axes minus the
         paper's NN, so it added nothing this does not.

BOTH ARE MEASURED INSIDE THE PAPER NN'S TRAINED RANGE, deliberately: eps0 within
+/-0.05*pi, |omega0| <= 12. Our full LHS envelope (eps0 within +/-pi/2, |omega0| <= 20)
drives vq to +/-1.14 while their released scaler saw +/-0.3, so numbers taken there
measure their network EXTRAPOLATING, not its accuracy. Plotting a method outside its
training range beside one inside it is not a comparison, whichever way it falls. Our own
full-envelope robustness is reported separately in notes.md as a property of our model.

NOT COMPARABLE WITH graphs/23. That figure measures error against the 100 us training
solver at omega0 in +/-2; this one measures against a 12.5 us fine-grid reference inside
their range. Same-looking axes, two different definitions of "theta RMS" -- putting them
on one plot would be the F59 mistake in a new costume.
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

import matplotlib.pyplot as plt
import numpy as np

from paths import GRAPHS
from speed_benchmark import head_to_head

STYLE = {"solver @50us":    ("dimgray",    "s"),
         "solver @100us":   ("k",          "s"),
         "their NN @50us":  ("tab:orange", "o"),
         "their NN @100us": ("tab:red",    "o"),
         "ours @100us":     ("tab:blue",   "D")}
# extra model groups passed with --model get these, in order
GROUP_STYLE = [("tab:purple", "^"), ("#6baed6", "D"), ("#08306b", "D"), ("tab:green", "v")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", nargs="?", default=None,
                   help="one checkpoint, the original single-point figure. Omit when using --model")
    p.add_argument("--model", action="append", default=[], metavar="LABEL=GLOB",
                   help="a model group scored seed by seed and drawn as a band, e.g. "
                        "'ours L3_w128=runs/famR_W40_*sp0_L3_w128.pth'. Repeatable. EVERY group "
                        "must be unlimited, fixed-gain physics -- the paper NN models nothing else, "
                        "so anything else on this axis is the different-physics comparison")
    p.add_argument("--n_runs", type=int, default=32)
    p.add_argument("--out", default="12_head_to_head.png")
    a = p.parse_args()

    models = None
    if a.model:
        models = {}
        for spec in a.model:
            label, pat = spec.split("=", 1)
            paths = sorted(glob.glob(pat))
            if not paths:
                raise SystemExit(f"--model {label!r}: no checkpoints match {pat}")
            models[label] = paths
            STYLE[label] = GROUP_STYLE[(len(models) - 1) % len(GROUP_STYLE)]
    elif a.ckpt is None:
        raise SystemExit("pass a checkpoint, or at least one --model LABEL=GLOB")

    # INSIDE THEIR TRAINING RANGE ONLY, deliberately. The full envelope drives vq to +/-1.14 while their
    # released model was trained on +/-0.3, so its numbers there measure extrapolation.
    # Plotting a method outside its training range next to one inside it is not a
    # comparison, whichever way the result falls. Our own full-envelope robustness
    # (1.04x degradation) is reported in notes.md as a property of OUR model alone.
    res = head_to_head(n_runs=a.n_runs, their_range=True, ckpt=a.ckpt, models=models)
    group = lambda v: v if isinstance(v, list) else [v]     # a seed band, or one point

    fig, (axt, ax) = plt.subplots(1, 2, figsize=(15.5, 5.8))

    # LEFT: error growth in time. This was figure 09's left panel, computed from a
    # separate run of `accuracy_benchmark` that did not include the paper's NN. `res`
    # already carries the full (n_runs, W*S) error array for every method, so the panel
    # is free here AND covers one more method than 09 did. 09 is retired.
    for k, v in res.items():
        c, m = STYLE[k]
        curves = np.stack([e.abs().mean(0).numpy() for e, _ in group(v)])   # (seeds, T)
        t = np.arange(curves.shape[1]) * (0.5 / curves.shape[1])
        n = f"  ({len(curves)} seeds)" if len(curves) > 1 else ""
        axt.semilogy(t[1:], np.median(curves, 0)[1:], color=c, lw=1.1, label=k + n)
        if len(curves) > 1:
            axt.fill_between(t[1:], curves.min(0)[1:], curves.max(0)[1:], color=c, alpha=.15, lw=0)
    # Every method starts from the same initial condition, so the error at t=0 is exactly
    # zero and a log axis would autoscale down to 1e-16 of empty space. Drop that sample
    # and floor the axis where the curves actually live.
    axt.set_ylim(bottom=1e-5)
    axt.set_xlabel("time [s]")
    axt.set_ylabel(r"$|\theta$ error$|$, mean over runs [rad]")
    axt.set_title("Does the error grow, or does it saturate?", fontsize=10)
    axt.grid(alpha=0.3, which="both"); axt.legend(fontsize=8)

    for k, v in res.items():
        c, m = STYLE[k]
        rs = np.array([float(e.pow(2).mean().sqrt()) for e, _ in group(v)])
        cs = np.array([ms for _, ms in group(v)])
        rms, ms = float(np.median(rs)), float(np.median(cs))
        if len(rs) > 1:      # the seed range, so no single checkpoint poses as the result
            ax.errorbar(ms, rms, yerr=[[rms - rs.min()], [rs.max() - rms]], fmt="none",
                        ecolor=c, elinewidth=2, capsize=5, zorder=2)
        # A LEGEND, not inline labels: the DeepONets and the paper NN @50us land within 1% of
        # each other in RMS, so annotations beside the points overprint into nothing legible.
        ax.scatter(ms, rms, s=160, color=c, marker=m, zorder=3,
                   label=f"{k}:  {rms:.3g} rad,  {ms:.0f} ms"
                         + (f"  ({len(rs)} seeds)" if len(rs) > 1 else ""))
    ax.legend(fontsize=7.8, loc="upper right", framealpha=.92)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.margins(0.34)
    ax.set_xlabel("compute [ms per simulated second]")
    ax.set_ylabel(r"$\theta$ RMS error vs a 12.5 $\mu$s reference [rad]")
    ax.set_title("Accuracy vs cost — down and left is better", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    extra = ("\nEvery model here is UNLIMITED physics with fixed Kp=25, Ki=300 — the paper NN "
             "models nothing else, so the limited/tunable flagship does not belong on this axis. "
             "Bars and bands = range over seeds." if models else "")
    fig.suptitle("Us vs the paper's NN vs a plain MLP vs the solver — error in time, and accuracy "
                 "against cost\n"
                 f"Test envelope: inside the paper NN's trained range "
                 f"(9 deg max phase error, p99|Vq| = 0.278 vs its 0.30 limit); "
                 f"{a.n_runs} runs x 0.5 s" + extra, fontsize=10)
    fig.tight_layout()
    out = GRAPHS / a.out
    GRAPHS.mkdir(exist_ok=True); fig.savefig(out, dpi=160)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
