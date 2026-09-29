"""The paper's data figures at IEEE single-column width (3.5 in), Times, vector PDF + PNG.

    python src/analysis/paper_figures.py            # ~1 min; writes graphs/paper/

Fig. 2  accuracy vs compute  -- UNLIMITED physics, fixed gains, inside the one-step NN's
        trained range. Values are graphs/12's legend (medians over 4 seeds for the
        networks), the numbers Section III-B quotes; nothing is re-timed here, because a
        new timing session moves the ms by a few percent and the text would drift.
Fig. 3  Omega error over 10 famO validation runs (the same 10 graphs/03 shows): the
        original 5k model without the anchor vs the flagship (famO40k s6, anchor trained
        in), on ONE y-scale so the difference is not hidden by autoscaling.
Fig. 4  gain sensitivity on LIMITED physics, from Hyperparameter_sweep/31_gain_sensitivity_limited.npz
        (graphs/31; omega0 +/-2, 12 runs per cell): tunable 5k, flagship, fixed 25/300.

Colour: categorical slots blue/orange/aqua (validated all-pairs, light) + neutral grey for
the solver; one single-hue sequential map (Blues) on one shared log scale for the heatmaps.
"""

# src/ on the path: these scripts live in src/analysis/ but import the pipeline
# modules (paths, PLL_Simulator, train_pll, sweep) that stay in src/. Running
# `python src/analysis/foo.py` puts src/analysis on sys.path, not src/.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from common_test import load_f32
from dataset_generator import Dataset_Creator
from paths import ROOT
from pll_infer import rollout, truth
from train_pll import prepare, group_split

OUT = ROOT / "graphs" / "paper"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e4e2"
COL_W = 3.5
ORIG = "runs/famO_W40_n5000_W40_F4_mf503_wp0.3_s0sp0_L3_w128_g.pth"
FLAG = "runs/famO40k_W40_n40000_W40_F4_mf503_wp0.3_s6sp0_L3_w128_ao_g.pth"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4, "axes.edgecolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK,
    "pdf.fonttype": 42, "ps.fonttype": 42,          # embedded TrueType -- IEEE PDF eXpress
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"-> {OUT / name}.pdf / .png")


def style(ax):
    ax.grid(True, which="major", color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def fig2():
    # (label, compute ms per simulated second, theta RMS rad, colour, marker, filled, label offset pts, ha)
    pts = [("Trapezoidal solver, 100 µs",  1109, 8.71e-4, INK2,   "s", True,  (0, 7),   "center"),
           ("Trapezoidal solver, 50 µs",   2191, 6.18e-4, INK2,   "s", False, (-7, 0),  "right"),
           ("One-step NN [1], 50 µs",        60, 8.85e-4, ORANGE, "o", True,  (0, 7),   "center"),
           ("One-step NN [1], 100 µs",       30, 8.24e-3, ORANGE, "o", False, (6, -1),  "left"),
           ("Plain MLP",                     23, 1.45e-3, AQUA,   "^", True,  (-6, 0),  "right"),
           ("Ours",                          27, 8.83e-4, BLUE,   "D", True,  (-7, -8), "right")]
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    ax.axhline(8.71e-4, color=INK2, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.annotate("100 µs noise floor", (5500, 8.71e-4), xytext=(0, -7), textcoords="offset points",
                ha="right", va="center", fontsize=6.5, color=INK2)
    for lab, x, y, c, m, filled, off, ha in pts:
        ax.scatter([x], [y], s=26, marker=m, color=c if filled else "white", edgecolor=c,
                   linewidth=1.1, zorder=3)
        ax.annotate(lab, (x, y), xytext=off, textcoords="offset points", ha=ha, va="center",
                    fontsize=6.5, color=INK)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(8, 6000); ax.set_ylim(4.5e-4, 1.5e-2)
    ax.set_xlabel("Compute [ms per simulated second], batch size 1")
    ax.set_ylabel(r"$\theta$ RMS error [rad]")
    style(ax)
    save(fig, "fig2_accuracy_vs_compute")


def fig3(n_show=10):
    data, meta = Dataset_Creator.load_dataset("famO_W40.npz")
    torch.set_default_dtype(torch.float32)
    prep, W = prepare(data), meta["W"]
    _, va = group_split(prep["run_id"], 0.15, 0)
    runs = sorted(set(prep["run_id"][va].tolist()))[:n_show]
    t = np.arange(W * meta["S"]) * meta["dt"]
    fig, axs = plt.subplots(2, 1, figsize=(COL_W, 2.6), sharex=True, sharey=True)
    for ax, path, col, ttl in [(axs[0], ORIG, INK2, "(a) Original model: 5k runs, no anchor"),
                               (axs[1], FLAG, BLUE, "(b) Final model: 40k runs, anchor trained in")]:
        model, ck = load_f32(ROOT / path)
        rms = []
        with torch.no_grad():
            for r in runs:
                _, om = rollout(model, ck, prep, r, W, W)
                _, om_t = truth(prep, r, W, W)
                e = (om - om_t).numpy()
                rms.append(np.sqrt(np.mean(e ** 2)))
                ax.plot(t, e, color=col, lw=0.55, alpha=0.75)
        ax.axhline(0, color=INK, lw=0.5)
        ax.set_title(ttl, loc="left", pad=2)
        ax.text(0.995, 0.06, f"mean RMS over these runs: {np.mean(rms):.4f} rad/s",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color=INK2)
        ax.set_ylabel(r"$\Omega$ error [rad/s]")
        style(ax)
        print(f"  {ttl}: mean Omega RMS over {n_show} runs {np.mean(rms):.3e}")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_xlim(0, t[-1])
    fig.align_ylabels(axs)
    save(fig, "fig3_anchor_omega_error")


def fig4():
    d = np.load(ROOT / "Hyperparameter_sweep" / "31_gain_sensitivity_limited.npz")
    kp, ki, Z = d["kp"], d["ki"], d["Z"]
    print("  panels in the npz:", list(d["labels"]))
    titles = ["(a) Tunable, 5k", "(b) Tunable, final", "(c) Fixed 25/300"]
    edges = lambda g: np.concatenate([[g[0] - (g[1] - g[0]) / 2], (g[1:] + g[:-1]) / 2,
                                      [g[-1] + (g[-1] - g[-2]) / 2]])
    norm = LogNorm(Z.min(), Z.max())
    fig, axs = plt.subplots(1, 3, figsize=(COL_W, 1.55), sharey=True,
                            gridspec_kw={"wspace": 0.12})
    for i, (ax, ttl) in enumerate(zip(axs, titles)):
        im = ax.pcolormesh(edges(kp), edges(ki), Z[i], cmap="Blues", norm=norm,
                           edgecolors="white", linewidth=0.3)
        ax.plot(25, 300, marker="*", ms=6, color=ORANGE, mec=INK, mew=0.4)
        ax.set_title(ttl, pad=2)
        ax.set_xticks([10, 25, 50]); ax.set_yticks([100, 300, 600])
        ax.set_xlabel(r"$K_p$", labelpad=1)
        ax.tick_params(length=2, pad=1.5)
        for s in ax.spines.values():
            s.set_visible(False)
    axs[0].set_ylabel(r"$K_i$", labelpad=1)
    cb = fig.colorbar(im, ax=list(axs), pad=0.02, fraction=0.05, aspect=14)
    cb.set_label(r"$\theta$ RMS [rad]", labelpad=2)
    cb.ax.tick_params(labelsize=6.5, length=2, width=0.5)
    cb.outline.set_linewidth(0.4)
    save(fig, "fig4_gain_sensitivity")


def fig5(n_test=150):
    """OPTIONAL example trajectory. The run is chosen by a fixed rule, not by eye: among
    famO's first 150 validation runs (the common test set), the LARGEST phase jump after
    which the TRUE PLL enters the +/-3 Hz limit. Its error rank among all 150 is printed,
    so the caption can say how typical it is."""
    data, meta = Dataset_Creator.load_dataset("famO_W40.npz")
    torch.set_default_dtype(torch.float32)
    prep, W, S, dt, L = prepare(data), meta["W"], meta["S"], meta["dt"], meta["freq_limit"]
    _, va = group_split(prep["run_id"], 0.15, 0)
    runs = sorted(set(prep["run_id"][va].tolist()))[:n_test]
    z = np.load(ROOT / "data" / "famO_W40.npz")
    kind, dist = z["fault_kind"][::W], z["disturbance"]
    t = np.arange(W * S) * dt
    cat = lambda key, r: torch.cat([prep[key][r * W + k] for k in range(W)]).numpy()

    def saturates_after_jump(r):
        u = cat("target_omega", r) + float(prep["kp"][r * W]) * cat("Vq", r)
        return np.any(np.abs(u[t >= dist[r, 3]]) > L)
    cand = [r for r in runs if kind[r] == 2 and saturates_after_jump(r)]
    r = max(cand, key=lambda q: abs(dist[q, 4]))
    t0, ang = dist[r, 3], np.degrees(dist[r, 4])
    kp, ki = float(prep["kp"][r * W]), float(prep["ki"][r * W])

    model, ck = load_f32(ROOT / FLAG)
    with torch.no_grad():
        rms = {q: float((rollout(model, ck, prep, q, W, W)[0] - truth(prep, q, W, W)[0]).pow(2).mean().sqrt())
               for q in runs}
        th_p = rollout(model, ck, prep, r, W, W)[0].double().numpy()
    th_t = truth(prep, r, W, W)[0].double().numpy()
    rank = sorted(rms.values()).index(rms[r]) + 1
    print(f"  run {r}: {ang:+.1f} deg jump at {t0:.3f} s, Kp {kp:.1f} Ki {ki:.0f}; "
          f"theta RMS {rms[r]:.2e} = rank {rank}/{len(runs)} (median {np.median(list(rms.values())):.2e})")

    wn = 2 * np.pi * 50
    dev = lambda th: th - th[0] - wn * t                      # angle minus the nominal ramp
    f = lambda th: (np.gradient(th, dt) - wn) / (2 * np.pi)   # PLL frequency deviation [Hz]
    fig, axs = plt.subplots(3, 1, figsize=(COL_W, 3.1), sharex=True,
                            gridspec_kw={"height_ratios": [1, 1, 0.75], "hspace": 0.18})
    # short labels; the caption defines them: dtheta = theta - theta0 - w_n t, df = PLL freq - 50 Hz
    for ax, (yt, yp, lab) in zip(axs[:2], [(dev(th_t), dev(th_p), r"$\Delta\theta$ [rad]"),
                                         (f(th_t), f(th_p), r"$\Delta f$ [Hz]")]):
        ax.plot(t, yt, color=INK, lw=1.0, label="Trapezoidal solver")
        ax.plot(t, yp, color=BLUE, lw=1.0, ls=(0, (3, 1.5)), label="DeepONet (final)")
        ax.set_ylabel(lab)
    for lim in (-3, 3):
        axs[1].axhline(lim, color=INK2, lw=0.6, ls=(0, (1, 1.5)))
    axs[1].annotate("±3 Hz limit", (0.495, 3), xytext=(0, 2), textcoords="offset points",
                    ha="right", va="bottom", fontsize=6.5, color=INK2)
    axs[2].plot(t, 1e3 * (th_p - th_t), color=BLUE, lw=0.8)
    axs[2].axhline(0, color=INK, lw=0.5)
    axs[2].set_ylabel(r"$\theta$ error [mrad]")
    axs[2].set_xlabel("Time [s]")
    for ax in axs:
        ax.axvline(t0, color=ORANGE, lw=0.8, alpha=0.8)
        style(ax)
    axs[0].annotate(f"{ang:+.0f}° phase jump", (t0, 1), xycoords=("data", "axes fraction"),
                    xytext=(3, -2), textcoords="offset points", ha="left", va="top",
                    fontsize=6.5, color=ORANGE)
    axs[0].legend(loc="upper right", frameon=False, handlelength=2.2)
    axs[2].set_xlim(0, t[-1])
    fig.align_ylabels(axs)
    save(fig, "fig5_example_trajectory")


if __name__ == "__main__":
    # python src/analysis/paper_figures.py [2 3 4 5] -- default: the three in the paper
    want = _sys.argv[1:] or ["2", "3", "4"]
    for k in want:
        {"2": fig2, "3": fig3, "4": fig4, "5": fig5}[k]()
