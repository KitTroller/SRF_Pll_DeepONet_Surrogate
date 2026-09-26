"""Is the tunable-gain model usable NEAR the co-simulation's operating point? graphs/Tunable_Kp_Ki_tests/03.

    python src/gain_sensitivity.py runs/famK_W40_n5000_W40_F4_mf503_wp0.3_s0sp0_g.pth

F57 measured the average cost of making Kp and Ki inputs: 3.5x on deployed angle error.
But an average over a box spanning zeta = 0.20 to 2.50 is not what they need -- they run at
Kp=25, Ki=300. If the model is accurate near there and only degrades at the corners, 3.5x
is pessimistic FOR HIM. If it is uniformly 3.5x, it is not.

Method: for each (Kp, Ki) cell, simulate the truth at that cell's gains with the SAME
initial conditions in every cell, then roll the surrogate recurrently over all W windows
feeding it those gains. The metric is the same `rollout_full_rms` reported everywhere
else -- vs the solver at the model's own dt, so this isolates network error from
discretisation.

Initial conditions use omega0 in +/-2, i.e. the warm-co-simulation regime, not the full
acquisition envelope. That is the honest test: how good is it where it is actually run.
"""

# src/ on the path: these scripts live in src/analysis/ but import the pipeline
# modules (paths, PLL_Simulator, train_pll, sweep) that stay in src/. Running
# `python src/analysis/foo.py` puts src/analysis on sys.path, not src/.
# Same pattern as hpc/generate_family.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import argparse
from functools import lru_cache

import numpy as np
import torch

import PLL_Simulator as PS
from PLL_Simulator import PLLSimulator
from paths import GRAPHS, ROOT
from common_test import load_f32, truth as _truth, rollout_rms
from train_pll import OMEGA_BASE

KP_GRID = [10.0, 18.0, 25.0, 33.0, 41.0, 50.0]
KI_GRID = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
THEIRS = (25.0, 300.0)


@lru_cache(maxsize=None)
def truth(kp, ki, n_runs, W, S, dt, seed=0):
    """Ground truth at ONE (Kp,Ki), with the same ICs in every cell.

    Thin shim over common_test.truth, kept so the (W, S, dt) call signature used
    throughout this file still works. Cached: every model is scored on the same cells,
    and the limited solve is the slow part."""
    return _truth(kp, ki, n_runs, seed=seed, dt=dt, n=W * S)


def rollout_err(model, ck, kp, ki, n_runs, W, S, dt):
    """THE common-test harness. Fresh trajectories at these gains, full recurrent
    rollout, mean per-run theta RMS -- comparable across families in a way that any
    per-family `val_th` is not (F59/F61)."""
    Va, Vb, Vc, th_t, om_t = truth(kp, ki, n_runs, W, S, dt)
    return rollout_rms(model, ck, (Va, Vb, Vc), th_t, om_t, kp, ki)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", nargs="+", help="one or more TUNABLE-gain checkpoints, one panel each")
    p.add_argument("--labels", nargs="+", default=None, help="panel titles, in ckpt order then --fixed")
    p.add_argument("--n_runs", type=int, default=6)
    p.add_argument("--fixed", default=None,
                   help="a FIXED-gain checkpoint to compare against. It was trained at one "
                        "tuning only, so it should be good at 25/300 and bad elsewhere -- "
                        "which is exactly the argument for making the gains inputs.")
    p.add_argument("--out", default="Tunable_Kp_Ki_tests/03_gain_sensitivity.png",
                   help="under graphs/. The default is the UNLIMITED August bundle's figure -- "
                        "give limited models their own file")
    a = p.parse_args()
    path = lambda c: ROOT / c if not c.startswith("/") else c
    names = a.ckpt + ([a.fixed] if a.fixed else [])
    labels = a.labels or ([f"Kp, Ki as INPUTS ({_Path(c).stem.split('_W')[0]})" for c in a.ckpt]
                          + ([f"fixed Kp=25, Ki=300 ({_Path(a.fixed).stem.split('_W')[0]})"] if a.fixed else []))
    # The truth must be the physics the models were trained on. common_test.truth builds its
    # PLLSimulator from PLL_Simulator's module-level constants, whose freq_limit is null, so
    # without this every limited model would be scored against UNLIMITED truth.
    limits = {load_f32(path(c))[1]["data_meta"].get("freq_limit") for c in names}
    if len(limits) > 1:
        raise SystemExit(f"these models were trained on different physics (freq_limit {limits})")
    PS.pll_constants.freq_limit = limits.pop()
    print(f"truth: freq_limit = {PS.pll_constants.freq_limit}")

    panels = []
    for c, lbl in zip(a.ckpt, labels):
        model, ck = load_f32(path(c))
        if not getattr(model, "n_extra", 0):
            raise SystemExit(f"{c} is not a gains model -- pass a checkpoint whose tag ends _g")
        m = ck["data_meta"]; W, S, dt = m["W"], m["S"], m["dt"]
        gr = m.get("gains", {})
        print(f"{c.split('/')[-1]}   W={W} S={S} dt={dt*1e6:.0f}us   trained on "
              f"Kp {gr.get('Kp')}  Ki {gr.get('Ki')}\n")

        Z = np.zeros((len(KI_GRID), len(KP_GRID)))
        for j, ki in enumerate(KI_GRID):
            for i, kp in enumerate(KP_GRID):
                Z[j, i] = rollout_err(model, ck, kp, ki, a.n_runs, W, S, dt)
            print(f"  Ki={ki:5.0f} : " + "  ".join(f"{Z[j,i]:.2e}" for i in range(len(KP_GRID))), flush=True)
        ref = rollout_err(model, ck, *THEIRS, a.n_runs, W, S, dt)
        print(f"\n  at THEIR gains Kp=25 Ki=300 : {ref:.3e} rad")
        print(f"  box median {np.median(Z):.3e}   box worst {Z.max():.3e} "
              f"(at Kp={KP_GRID[Z.argmax()%len(KP_GRID)]:.0f}, Ki={KI_GRID[Z.argmax()//len(KP_GRID)]:.0f})")
        print(f"  -> at their operating point the model is {np.median(Z)/ref:.2f}x BETTER than the box median\n")
        panels.append((lbl, Z, ref))
    if a.fixed:
        # load_checkpoint rebuilds the MLP with the CURRENT default dtype, and the solves
        # above leave it at float64 -- the weights are float32, so inference would die on
        # a dtype mismatch. Same trap as accuracy_benchmark documents.
        torch.set_default_dtype(torch.float32)
        fm, fck = load_f32(path(a.fixed))
        torch.set_default_dtype(torch.float64)
        Zf = np.zeros_like(Z)
        print("\n  fixed-gain model on the same grid (trained at Kp=25, Ki=300 ONLY):")
        for j, ki in enumerate(KI_GRID):
            for i, kp in enumerate(KP_GRID):
                Zf[j, i] = rollout_err(fm, fck, kp, ki, a.n_runs, W, S, dt)
            print(f"  Ki={ki:5.0f} : " + "  ".join(f"{Zf[j,i]:.2e}" for i in range(len(KP_GRID))), flush=True)
        reff = rollout_err(fm, fck, *THEIRS, a.n_runs, W, S, dt)
        print(f"\n  fixed model at their gains: {reff:.3e}   box median {np.median(Zf):.3e}"
              f"   -> degrades {np.median(Zf)/reff:.1f}x away from its tuning")
        # "one grid step away" = Kp 25, Ki 300 -> 200, the cell F60's 38x was quoted at
        step = (KI_GRID.index(200.0), KP_GRID.index(25.0))
        for lbl, Zp, rp in panels:
            print(f"  vs {lbl}: at (25,300) fixed/tunable = {reff/rp:.2f}x | one step away (25,200) "
                  f"= {Zf[step]/Zp[step]:.1f}x | box median = {np.median(Zf)/np.median(Zp):.1f}x")
        panels.append((labels[-1], Zf, reff))

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    # the numbers, so a figure-only redraw (e.g. at IEEE column width) needs no recompute
    np.savez(ROOT / "Hyperparameter_sweep" / (_Path(a.out).stem + ".npz"), kp=KP_GRID, ki=KI_GRID,
             labels=[t for t, _, _ in panels], Z=np.stack([z for _, z, _ in panels]),
             ref=[r for _, _, r in panels], n_runs=a.n_runs)
    vmin = min(z.min() for _, z, _ in panels); vmax = max(z.max() for _, z, _ in panels)
    fig, axs = plt.subplots(1, len(panels), figsize=(8.6 * len(panels), 6.2), squeeze=False)
    # half a cell of padding, so each cell is CENTRED on its grid value (the old extent ran
    # edge-to-edge through the first and last values: labels off-centre, edge ones clipped)
    dkp = (KP_GRID[-1] - KP_GRID[0]) / (len(KP_GRID) - 1) / 2
    dki = (KI_GRID[-1] - KI_GRID[0]) / (len(KI_GRID) - 1) / 2
    for ax, (ttl, Zp, rp) in zip(axs[0], panels):
        im = ax.imshow(Zp, origin="lower", cmap="viridis_r", norm=LogNorm(vmin, vmax),
                       extent=[KP_GRID[0] - dkp, KP_GRID[-1] + dkp, KI_GRID[0] - dki, KI_GRID[-1] + dki],
                       aspect="auto")
        for j, ki in enumerate(KI_GRID):
            for i, kp in enumerate(KP_GRID):
                ax.annotate(f"{Zp[j,i]:.1e}", (kp, ki), ha="center", va="center",
                            fontsize=7, color="w")
        ax.plot(*THEIRS, "*", ms=26, color="tab:red", mec="k", mew=1.2, zorder=5)
        ax.annotate(f"  their tuning\n  {rp:.2e} rad", THEIRS, color="tab:red", fontsize=9,
                    fontweight="bold", va="center")
        kk, ii = np.meshgrid(np.linspace(KP_GRID[0], KP_GRID[-1], 200),
                             np.linspace(KI_GRID[0], KI_GRID[-1], 200))
        cs = ax.contour(kk, ii, kk / (2 * np.sqrt(ii)), levels=[0.3, 0.5, 0.707, 1.0, 1.5],
                        colors="w", linewidths=0.8, alpha=0.6)
        ax.clabel(cs, fmt=r"$\zeta$=%.2f", fontsize=7)
        ax.set_xlabel("$K_p$"); ax.set_ylabel("$K_i$"); ax.set_title(ttl, fontsize=11)
    fig.colorbar(im, ax=axs[0].tolist(), label=r"deployed $\theta$ RMS over 0.5 s [rad]")
    phys = ("LIMITED physics (+/-3 Hz)" if PS.pll_constants.freq_limit is not None
            else "UNLIMITED physics")
    fig.suptitle(f"Where in the (Kp, Ki) box is each model accurate?   {phys}, "
                 r"$\omega_0 \in \pm2$ (warm co-simulation), "
                 f"{a.n_runs} runs per cell, same ICs in every cell, " r"$\zeta$ contours in white",
                 fontsize=11)
    out = GRAPHS / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
