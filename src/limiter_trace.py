"""One run, both physics, with the clamp visible. graphs/25.

    python src/limiter_trace.py

graphs/03 shows prediction vs truth on the UNLIMITED physics. This is its counterpart for
the limiter: the same initial conditions and the same grid waveform solved twice -- once
with the limiter and once without -- so the two columns differ only by the clamp.

The top row is the thing being clamped, `u = omega + Kp*Vq`, against the +/-L band. That
is what makes "saturation" visible at all: everywhere else in the project it is invisible,
because only ~4% of windows touch it and theta itself looks perfectly ordinary.
"""
import argparse
import glob

import numpy as np
import torch

import PLL_Simulator as PS
from common_test import load_f32, rollout
from paths import graphs as _graphs
from train_pll import OMEGA_BASE

DT, N, W, S = 100e-6, 5000, 40, 125
LIMIT = 18.8496


def solve(limit, n_runs=12, seed=0):
    PS.pll_constants.freq_limit = limit
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    sim = PS.PLLSimulator(dt=DT)
    sim.N, sim.n_runs = N, n_runs
    sim.t = (torch.arange(N) * DT).reshape(1, N)
    ga = (torch.rand(n_runs, 1) * 2 - 1) * torch.pi
    fo = (torch.rand(n_runs, 1) * 2 - 1) * 0.2
    ao = (torch.rand(n_runs, 1) * 2 - 1) * 0.05
    th0 = ga.squeeze(-1) + (torch.rand(n_runs) * 2 - 1) * 0.5 * torch.pi
    th0 = (th0 + torch.pi) % (2 * torch.pi) - torch.pi
    om0 = (torch.rand(n_runs) * 2 - 1) * 20.0
    Va, Vb, Vc = sim._grid_phases(fo, ao, ga)
    with torch.no_grad():
        th, om, _, Vq = sim.simulate_batch(Va, Vb, Vc, th0, om0)[:4]
    return Va, Vb, Vc, th, om, om + sim.physics.Kp * Vq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="25_limiter_trace.png")
    a = p.parse_args()

    lim = solve(LIMIT)
    unl = solve(None)
    PS.pll_constants.freq_limit = None
    assert torch.equal(lim[0], unl[0]), "grid waveforms must be identical between the two"

    # the most-saturated run, so the effect is legible rather than a single blip
    sat_w = (lim[5].reshape(-1, W, S).abs().max(2).values > LIMIT)
    r = int(sat_w.sum(1).argmax())
    print(f"run {r}: {int(sat_w[r].sum())} of {W} windows saturate")

    ck_lim = sorted(q for q in glob.glob("runs/famN_W40_*sp0.pth") if "_L" not in q)[0]
    ck_unl = sorted(q for q in glob.glob("runs/famR_W40_*sp0.pth") if "_L" not in q)[0]

    import matplotlib.pyplot as plt
    t = np.arange(N) * DT
    fig, ax = plt.subplots(3, 2, figsize=(15, 9), sharex=True,
                           gridspec_kw=dict(height_ratios=[1.1, 2, 1.3]))

    for c, (dat, ckpt, L, name) in enumerate((
            (unl, ck_unl, None, "famR — NO limiter"),
            (lim, ck_lim, LIMIT, "famN — limiter at $\\omega_0 \\pm 2\\pi\\cdot3$"))):
        Va, Vb, Vc, th_t, om_t, u = dat
        m, ck = load_f32(ckpt)
        pth, _ = rollout(m, ck, (Va[r], Vb[r], Vc[r]), th_t[r, 0].float(), om_t[r, 0].float())
        torch.set_default_dtype(torch.float64)
        pth = pth.double().numpy()
        ramp = th_t[r, 0].item() + OMEGA_BASE * t
        err = np.abs(pth - th_t[r].numpy())
        sat = (u[r].reshape(W, S).abs().max(1).values > LIMIT).numpy() if L else np.zeros(W, bool)

        for row in range(3):                       # shade the saturated windows everywhere
            for k in np.where(sat)[0]:
                ax[row, c].axvspan(k * S * DT, (k + 1) * S * DT, color="tab:red", alpha=.13, lw=0)

        ax[0, c].plot(t, u[r].numpy(), "k", lw=.9)
        if L:
            ax[0, c].axhline(L, color="tab:red", ls="--", lw=1.2)
            ax[0, c].axhline(-L, color="tab:red", ls="--", lw=1.2)
            ax[0, c].text(0.42, L * 1.08, "clamp $\\pm$18.85 rad/s", color="tab:red", fontsize=8)
        ax[0, c].set_title(name, fontsize=11)
        ax[0, c].set_ylabel("$u=\\omega+K_p V_q$ [rad/s]")

        ax[1, c].plot(t, th_t[r].numpy() - ramp, "k", lw=2.2, label="truth (trapezoid)")
        ax[1, c].plot(t, pth - ramp, "tab:blue" if c == 0 else "tab:red", lw=1.1,
                      label="surrogate, 40 recurrent handovers")
        ax[1, c].set_ylabel("$\\theta-(\\theta_0+\\omega_{base}t)$ [rad]")
        ax[1, c].legend(fontsize=8, loc="best")

        ax[2, c].semilogy(t, err, "tab:blue" if c == 0 else "tab:red", lw=.9)
        ax[2, c].set_ylabel("$|\\theta$ error$|$ [rad]")
        ax[2, c].set_xlabel("time [s]")
        for row in range(3):
            ax[row, c].grid(alpha=.3)

    lo = min(ax[2, 0].get_ylim()[0], ax[2, 1].get_ylim()[0])
    hi = max(ax[2, 0].get_ylim()[1], ax[2, 1].get_ylim()[1])
    for c in (0, 1):
        ax[2, c].set_ylim(lo, hi)                  # same axis or the comparison is a lie
        ax[1, c].set_ylim(*ax[1, 0].get_ylim())

    fig.suptitle("Same initial conditions, same grid voltage, solved twice.  "
                 "Red bands = windows where the clamp is active.\n"
                 "The angle looks ordinary throughout — the limiter is only visible in "
                 "the top row, which is why the aggregate metric misses it.", fontsize=11.5)
    fig.tight_layout()
    out = _graphs(a.out); fig.savefig(out, dpi=140)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
