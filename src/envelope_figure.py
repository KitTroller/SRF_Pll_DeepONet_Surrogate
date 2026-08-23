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
import argparse

import matplotlib.pyplot as plt
import numpy as np

from paths import GRAPHS
from speed_benchmark import head_to_head

STYLE = {"solver @50us":    ("dimgray",    "s"),
         "solver @100us":   ("k",          "s"),
         "their NN @50us":  ("tab:orange", "o"),
         "their NN @100us": ("tab:red",    "o"),
         "ours @100us":     ("tab:blue",   "D")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--n_runs", type=int, default=32)
    a = p.parse_args()

    # INSIDE THEIR TRAINING RANGE ONLY, deliberately. The full envelope drives vq to +/-1.14 while their
    # released model was trained on +/-0.3, so its numbers there measure extrapolation.
    # Plotting a method outside its training range next to one inside it is not a
    # comparison, whichever way the result falls. Our own full-envelope robustness
    # (1.04x degradation) is reported in notes.md as a property of OUR model alone.
    res = head_to_head(n_runs=a.n_runs, their_range=True, ckpt=a.ckpt)

    fig, (axt, ax) = plt.subplots(1, 2, figsize=(14.5, 5.4))

    # LEFT: error growth in time. This was figure 09's left panel, computed from a
    # separate run of `accuracy_benchmark` that did not include the paper's NN. `res`
    # already carries the full (n_runs, W*S) error array for every method, so the panel
    # is free here AND covers one more method than 09 did. 09 is retired.
    for k, (e, ms) in res.items():
        c, m = STYLE[k]
        t = np.arange(e.shape[1]) * (0.5 / e.shape[1])
        axt.semilogy(t[1:], e.abs().mean(0).numpy()[1:], color=c, lw=1.1, label=k)
    # Every method starts from the same initial condition, so the error at t=0 is exactly
    # zero and a log axis would autoscale down to 1e-16 of empty space. Drop that sample
    # and floor the axis where the curves actually live.
    axt.set_ylim(bottom=1e-5)
    axt.set_xlabel("time [s]")
    axt.set_ylabel(r"$|\theta$ error$|$, mean over runs [rad]")
    axt.set_title("Does the error grow, or does it saturate?", fontsize=10)
    axt.grid(alpha=0.3, which="both"); axt.legend(fontsize=8)

    for k, (e, ms) in res.items():
        rms = float(e.pow(2).mean().sqrt())
        c, m = STYLE[k]
        ax.scatter(ms, rms, s=160, color=c, marker=m, zorder=3)
        ax.annotate(f"{k}\n{rms:.3g} rad", (ms, rms), textcoords="offset points",
                    xytext=(9, 9), fontsize=8.5)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.margins(0.34)
    ax.set_xlabel("compute [ms per simulated second]")
    ax.set_ylabel(r"$\theta$ RMS error vs a 12.5 $\mu$s reference [rad]")
    ax.set_title("Accuracy vs cost — down and left is better", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    fig.suptitle("Us vs the paper's NN vs the solver — error in time, and accuracy against cost\n"
                 f"Test envelope: inside the paper NN's trained range "
                 f"(9 deg max phase error, p99|Vq| = 0.278 vs its 0.30 limit); "
                 f"{a.n_runs} runs x 0.5 s", fontsize=10)
    fig.tight_layout()
    out = GRAPHS / "12_head_to_head.png"
    GRAPHS.mkdir(exist_ok=True); fig.savefig(out, dpi=160)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
