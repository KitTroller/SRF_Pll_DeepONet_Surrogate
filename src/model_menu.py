"""The model menu for the EMT co-simulation. Writes into graphs/Tunable_Kp_Ki_tests/.

    python src/model_menu.py

He has to pick three things, and each has a measured price:

    window length   W=40 (12.5 ms) or W=20 (25 ms). W=20 HALVES the network calls per
                    simulated second, which is a real speed lever because inference is
                    overhead-bound, not FLOP-bound (F25).
    omega range     wide (+/-20, covers acquisition) or narrow (+/-2, the co-simulation regime). Their
                    co-simulation never leaves |omega| < 0.15, and only 6.8% of the wide
                    family's windows are in that band.
    gains           fixed Kp/Ki, or Kp/Ki as network INPUTS so they can be retuned without a
                    retrain.

Cost per simulated second is `calls/s x cost per call`, and calls/s = 1/(S*dt) -- it does
NOT depend on the network size, which is why W is the lever and hidden_dim is not.
"""
import collections
import glob
import json
import statistics as st

import numpy as np

from paths import GRAPHS, ROOT

OUT = GRAPHS / "Tunable_Kp_Ki_tests"

# label -> (results dir, W, wide/narrow, fixed/gains)
VARIANTS = [
    ("W40 wide  fixed",   "sweeps_famD",     40, "wide",   "fixed"),
    ("W40 narrow fixed",  "sweeps_famH_W40", 40, "narrow", "fixed"),
    ("W20 wide  fixed",   "sweeps_famI_W20", 20, "wide",   "fixed"),
    ("W20 narrow fixed",  "sweeps_famH_W20", 20, "narrow", "fixed"),
    ("W40 wide  GAINS",   "sweeps_famJ_W40", 40, "wide",   "gains"),
    ("W20 wide  GAINS",   "sweeps_famJ_W20", 20, "wide",   "gains"),
    ("W40 narrow GAINS",  "sweeps_famK_W40", 40, "narrow", "gains"),
    ("W20 narrow GAINS",  "sweeps_famK_W20", 20, "narrow", "gains"),
]
DT = 100e-6                                   # all of these are 5000 sensors over 0.5 s


def load():
    out = []
    for lab, d, W, om, g in VARIANTS:
        recs = [json.loads(open(f).read()) for f in glob.glob(str(ROOT / "Hyperparameter_sweep" / d / "*.json"))]
        recs = [r for r in recs if r.get("status", "ok") == "ok"]
        if not recs:
            print(f"  (skipping {lab}: no records yet)")
            continue
        S = recs[0]["S"]
        out.append(dict(label=lab, W=W, omega=om, gains=g, n=len(recs), S=S,
                        calls=1.0 / (S * DT),                       # network calls per simulated second
                        th=[r["rollout_full_rms"] for r in recs],
                        om=[r["val_om"] for r in recs],
                        thv=[r["val_th"] for r in recs]))
    return out


def fig_menu(V):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    mk = {"fixed": "o", "gains": "s"}
    co = {"wide": "tab:blue", "narrow": "tab:green"}
    for v in V:
        y = st.median(v["th"])
        ax.errorbar(v["calls"], y, yerr=[[y - min(v["th"])], [max(v["th"]) - y]],
                    fmt=mk[v["gains"]], color=co[v["omega"]], ms=11, capsize=4, lw=1.4)
        ax.annotate(f"  {v['label']}\n  {y:.2e} rad  (n={v['n']})", (v["calls"], y),
                    fontsize=7.5, va="center")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.margins(0.3)
    ax.set_xlabel("network calls per simulated second   (lower = cheaper; W=20 halves it)")
    ax.set_ylabel(r"deployed $\theta$ RMS over 0.5 s [rad]   (bar = seed range)")
    ax.grid(alpha=0.3, which="both")
    h = [plt.Line2D([], [], ls="", marker="o", color="k", label="fixed Kp/Ki"),
         plt.Line2D([], [], ls="", marker="s", color="k", label="Kp/Ki as inputs"),
         plt.Line2D([], [], ls="", marker="o", color="tab:blue", label=r"wide $\omega_0$ ($\pm$20)"),
         plt.Line2D([], [], ls="", marker="o", color="tab:green", label=r"narrow $\omega_0$ ($\pm$2)")]
    ax.legend(handles=h, fontsize=8, loc="upper left")
    ax.set_title("The model menu — down and left is better\n"
                 "All at dt = 100 us. Deployed = 0.5 s, recurrent, no ground truth.", fontsize=10)
    fig.tight_layout(); OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "01_model_menu.png", dpi=160); print(f"-> {OUT/'01_model_menu.png'}")


def fig_theta_omega(V):
    """theta and omega separately -- omega was the visibly weak output in the first closed-loop test,
    and the two do not move together across these variants."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.0))
    y = np.arange(len(V))
    co = ["tab:green" if v["omega"] == "narrow" else "tab:blue" for v in V]
    hatch = ["///" if v["gains"] == "gains" else "" for v in V]
    for a, key, lab, ttl in [(ax[0], "th", r"deployed $\theta$ RMS [rad]", r"$\theta$ — the angle the Park transform needs"),
                             (ax[1], "om", r"val MSE $\omega$ [rad$^2$/s$^2$]", r"$\omega$ — the visibly weak output in the first co-sim test")]:
        med = [st.median(v[key]) for v in V]
        for i, v in enumerate(V):
            a.barh(y[i], med[i], 0.7, color=co[i], hatch=hatch[i], edgecolor="k", lw=0.6)
            a.annotate(f"  {med[i]:.2e}", (med[i], y[i]), va="center", fontsize=7.5)
        a.set_xscale("log"); a.set_yticks(y); a.set_yticklabels([v["label"] for v in V], fontsize=8)
        a.invert_yaxis(); a.set_xlabel(lab); a.set_title(ttl, fontsize=10)
        a.grid(alpha=0.3, axis="x", which="both"); a.margins(x=0.3)
    fig.suptitle("Green = narrow omega range, blue = wide. Hatched = Kp/Ki as inputs.", fontsize=10)
    fig.tight_layout(); OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "02_theta_omega.png", dpi=160); print(f"-> {OUT/'02_theta_omega.png'}")


if __name__ == "__main__":
    V = load()
    if not V:
        raise SystemExit("no records found")
    print(f"\n{'variant':22s} {'n':>2s} {'calls/s':>8s} {'deployed theta':>16s} {'val_om':>12s}")
    for v in V:
        print(f"{v['label']:22s} {v['n']:2d} {v['calls']:8.0f} {st.median(v['th']):16.3e} {st.median(v['om']):12.3e}")
    fig_menu(V); fig_theta_omega(V)
