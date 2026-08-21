"""Is the tunable-gain model usable NEAR Rahul's operating point? graphs/rahul/03.

    python src/gain_sensitivity.py runs/famK_W40_n5000_W40_F4_mf503_wp0.3_s0sp0_g.pth

F57 measured the average cost of making Kp and Ki inputs: 3.5x on deployed angle error.
But an average over a box spanning zeta = 0.20 to 2.50 is not what he needs -- he runs at
Kp=25, Ki=300. If the model is accurate near there and only degrades at the corners, 3.5x
is pessimistic FOR HIM. If it is uniformly 3.5x, it is not.

Method: for each (Kp, Ki) cell, simulate the truth at that cell's gains with the SAME
initial conditions in every cell, then roll the surrogate recurrently over all W windows
feeding it those gains. The metric is the same `rollout_full_rms` reported everywhere
else -- vs the solver at the model's own dt, so this isolates network error from
discretisation.

Initial conditions use omega0 in +/-2, i.e. the warm-co-simulation regime, not the full
acquisition envelope. That is the honest test: how good is it where he actually runs it.
"""
import argparse

import numpy as np
import torch

from PLL_Simulator import PLLSimulator
from paths import GRAPHS, ROOT
from pll_infer import predict_window
from train_pll import load_checkpoint, OMEGA_BASE

KP_GRID = [10.0, 18.0, 25.0, 33.0, 41.0, 50.0]
KI_GRID = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
THEIRS = (25.0, 300.0)


def truth(kp, ki, n_runs, W, S, dt, seed=0):
    """Ground truth at ONE (Kp,Ki), with the same ICs in every cell."""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)                      # identical ICs and noise across cells
    N = W * S
    sim = PLLSimulator(dt=dt)
    sim.N, sim.n_runs = N, n_runs
    sim.t = (torch.arange(N) * dt).reshape(1, N)
    sim.physics.Kp = torch.full((n_runs,), kp, dtype=torch.float64)
    sim.physics.Ki = torch.full((n_runs,), ki, dtype=torch.float64)
    ga = (torch.rand(n_runs, 1) * 2 - 1) * torch.pi
    fo = (torch.rand(n_runs, 1) * 2 - 1) * 0.2
    ao = (torch.rand(n_runs, 1) * 2 - 1) * 0.05
    th0 = ga.squeeze(-1) + (torch.rand(n_runs) * 2 - 1) * 0.5 * torch.pi
    th0 = (th0 + torch.pi) % (2 * torch.pi) - torch.pi
    om0 = (torch.rand(n_runs) * 2 - 1) * 2.0     # HIS regime, not the full +/-20
    Va, Vb, Vc = sim._grid_phases(fo, ao, ga)
    with torch.no_grad():
        th, om = sim.simulate_batch(Va, Vb, Vc, th0, om0, scheme="trapezoid")[:2]
    return Va, Vb, Vc, th, om


def rollout_err(model, ck, kp, ki, n_runs, W, S, dt):
    Va, Vb, Vc, th_t, om_t = truth(kp, ki, n_runs, W, S, dt)
    torch.set_default_dtype(torch.float32)
    t = ck["t_local"]
    t_ext = torch.cat([t, t[-1:] + float(t[1] - t[0])])
    errs = []
    for r in range(n_runs):
        th0, om0 = th_t[r, 0].float(), om_t[r, 0].float()
        pred = []
        want = bool(getattr(model, "n_extra", 0))     # fixed-gain models take no kp/ki
        for k in range(W):
            sl = slice(k * S, (k + 1) * S)
            p, o = predict_window(model, ck, th0, om0, Va[r, sl].float(),
                                  Vb[r, sl].float(), Vc[r, sl].float(), t_ext,
                                  kp if want else None, ki if want else None)
            pred.append(p[:-1]); th0, om0 = p[-1], o[-1]        # the handover
        e = torch.cat(pred).double() - th_t[r]
        errs.append(float(e.pow(2).mean().sqrt()))
    torch.set_default_dtype(torch.float64)
    return float(np.mean(errs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--n_runs", type=int, default=6)
    p.add_argument("--fixed", default=None,
                   help="a FIXED-gain checkpoint to compare against. It was trained at one "
                        "tuning only, so it should be good at 25/300 and bad elsewhere -- "
                        "which is exactly the argument for making the gains inputs.")
    a = p.parse_args()
    model, ck = load_checkpoint(ROOT / a.ckpt if not a.ckpt.startswith("/") else a.ckpt)
    if not getattr(model, "n_extra", 0):
        raise SystemExit("this is not a gains model -- pass a checkpoint whose tag ends _g")
    m = ck["data_meta"]; W, S, dt = m["W"], m["S"], m["dt"]
    gr = m.get("gains", {})
    print(f"{a.ckpt.split('/')[-1]}   W={W} S={S} dt={dt*1e6:.0f}us   trained on "
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
    print(f"  -> at their operating point the model is {np.median(Z)/ref:.2f}x BETTER than the box median")

    panels = [("Kp, Ki as INPUTS (famK)", Z, ref)]
    if a.fixed:
        # load_checkpoint rebuilds the MLP with the CURRENT default dtype, and the solves
        # above leave it at float64 -- the weights are float32, so inference would die on
        # a dtype mismatch. Same trap as accuracy_benchmark documents.
        torch.set_default_dtype(torch.float32)
        fm, fck = load_checkpoint(ROOT / a.fixed if not a.fixed.startswith("/") else a.fixed)
        torch.set_default_dtype(torch.float64)
        Zf = np.zeros_like(Z)
        print("\n  fixed-gain model on the same grid (trained at Kp=25, Ki=300 ONLY):")
        for j, ki in enumerate(KI_GRID):
            for i, kp in enumerate(KP_GRID):
                Zf[j, i] = rollout_err(fm, fck, kp, ki, a.n_runs, W, S, dt)
            print(f"  Ki={ki:5.0f} : " + "  ".join(f"{Zf[j,i]:.2e}" for i in range(len(KP_GRID))), flush=True)
        reff = rollout_err(fm, fck, *THEIRS, a.n_runs, W, S, dt)
        print(f"\n  fixed model at their gains: {reff:.3e}   box median {np.median(Zf):.3e}")
        print(f"  -> fixed model degrades {np.median(Zf)/reff:.1f}x away from its tuning; "
              f"gains model only {np.median(Z)/ref:.2f}x")
        panels.append(("fixed Kp=25, Ki=300 (famH)", Zf, reff))

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    vmin = min(z.min() for _, z, _ in panels); vmax = max(z.max() for _, z, _ in panels)
    fig, axs = plt.subplots(1, len(panels), figsize=(8.6 * len(panels), 6.2), squeeze=False)
    for ax, (ttl, Zp, rp) in zip(axs[0], panels):
        im = ax.imshow(Zp, origin="lower", cmap="viridis_r", norm=LogNorm(vmin, vmax),
                       extent=[KP_GRID[0], KP_GRID[-1], KI_GRID[0], KI_GRID[-1]], aspect="auto")
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
    fig.suptitle("Where in the (Kp, Ki) box is each model accurate?   "
                 r"$\omega_0 \in \pm2$ (warm co-simulation), $\zeta$ contours in white",
                 fontsize=11)
    out = GRAPHS / "rahul"; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "03_gain_sensitivity.png", dpi=160)
    print(f"\n-> {out/'03_gain_sensitivity.png'}")


if __name__ == "__main__":
    main()
