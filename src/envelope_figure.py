"""One figure that subsumes 09 and 12: accuracy vs cost, on BOTH test envelopes.

    python src/envelope_figure.py runs/famD_W40_n5000_W40_F4_mf503_wp0.3_s0sp0.pth

Why two panels rather than two figures. The two surrogates are trained for different
operating ranges, and that IS the comparison:

  inside their training range   eps0 within +/-0.05*pi, |omega0| <= 12  -- inside BOTH training envelopes.
              Their NN's scaler says it saw vq in +/-0.3, which only holds near lock.
  full        eps0 within +/-pi/2, |omega0| <= 20     -- our LHS range. Drives vq to
              +/-1.14, ~4x outside anything their network was trained on, so its numbers
              here measure EXTRAPOLATION, not accuracy. Marked as such.

Read the left panel for "are the two networks equivalent where both are valid" and the
right panel for "what does covering the wider envelope cost each of them".
"""
import argparse

import matplotlib.pyplot as plt

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

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for k, (e, ms) in res.items():
        rms = float(e.pow(2).mean().sqrt())
        c, m = STYLE[k]
        ax.scatter(ms, rms, s=160, color=c, marker=m, zorder=3)
        ax.annotate(f"{k}\n{rms:.3g} rad", (ms, rms), textcoords="offset points",
                    xytext=(9, 9), fontsize=8.5)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.margins(0.34)
    ax.set_xlabel("compute [ms per simulated second]")
    ax.set_ylabel(r"$\theta$ RMS error vs a 12.5 $\mu$s reference [rad]")
    ax.grid(alpha=0.3, which="both")
    fig.suptitle("Accuracy vs cost — down and left is better\n"
                 f"Test envelope: inside the paper NN's trained range "
                 f"(9 deg max phase error, p99|Vq| = 0.278 vs its 0.30 limit); "
                 f"{a.n_runs} runs x 0.5 s", fontsize=10)
    fig.tight_layout()
    out = GRAPHS / "12_head_to_head.png"
    GRAPHS.mkdir(exist_ok=True); fig.savefig(out, dpi=160)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
