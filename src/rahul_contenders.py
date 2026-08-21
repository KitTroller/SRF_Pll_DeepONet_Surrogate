"""Prediction vs truth for every model offered to Rahul. graphs/rahul/04, 05, 06.

    python src/rahul_contenders.py

Eight contenders, four per window length, all at dt = 100 us:

    wide   omega0 in +/-20  -- covers PLL acquisition
    narrow omega0 in +/-2   -- his warm-co-simulation regime
    fixed  Kp=25, Ki=300 baked in
    gains  Kp, Ki as network INPUTS

Every model is rolled out RECURRENTLY over the SAME 0.5 s trajectory -- its own state fed
forward, no ground truth anywhere -- so the curves are directly comparable and the window
length is the only thing that differs in how they are driven.

Figure 06 is the one that justifies the gains feature: the same fixed-gain and tunable
models evaluated at three controller tunings. A model trained at one tuning is not
approximately right at another; it is wrong by two orders of magnitude.
"""
import numpy as np
import torch

from PLL_Simulator import PLLSimulator
from paths import GRAPHS, ROOT
from pll_infer import predict_window
from train_pll import load_checkpoint, OMEGA_BASE

OUT = GRAPHS / "rahul"
DT, N = 100e-6, 5000
THEIRS = (25.0, 300.0)

MODELS = [  # label, checkpoint, colour
    ("wide  fixed",  "famD_W40_n5000_W40_F4_mf503_wp0.3_s1sp0.pth",   "tab:blue"),
    ("narrow fixed", "famH_W40_n5000_W40_F4_mf503_wp0.3_s0sp0.pth",   "tab:green"),
    ("wide  GAINS",  "famJ_W40_n5000_W40_F4_mf503_wp0.3_s0sp0_g.pth", "tab:orange"),
    ("narrow GAINS", "famK_W40_n5000_W40_F4_mf503_wp0.3_s0sp0_g.pth", "tab:red"),
    ("wide  fixed",  "famI_W20_n5000_W20_F4_mf503_wp0.3_s0sp0.pth",   "tab:blue"),
    ("narrow fixed", "famH_W20_n5000_W20_F4_mf503_wp0.3_s0sp0.pth",   "tab:green"),
    ("wide  GAINS",  "famJ_W20_n5000_W20_F4_mf503_wp0.3_s0sp0_g.pth", "tab:orange"),
    ("narrow GAINS", "famK_W20_n5000_W20_F4_mf503_wp0.3_s2sp0_g.pth", "tab:red"),
]


def truth(kp, ki, seed=3):
    """ONE 0.5 s trajectory. Same ICs regardless of gains, so the only thing that changes
    between tunings is the loop's response."""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    sim = PLLSimulator(dt=DT)
    sim.N, sim.n_runs = N, 1
    sim.t = (torch.arange(N) * DT).reshape(1, N)
    sim.physics.Kp = torch.tensor([kp]); sim.physics.Ki = torch.tensor([ki])
    ga = torch.tensor([[0.4]]); fo = torch.tensor([[0.12]]); ao = torch.tensor([[0.02]])
    th0 = torch.tensor([0.4 + 0.9])            # ~0.9 rad of initial phase error
    om0 = torch.tensor([1.5])                  # inside the narrow range, so all 8 are valid
    Va, Vb, Vc = sim._grid_phases(fo, ao, ga)
    with torch.no_grad():
        th, om = sim.simulate_batch(Va, Vb, Vc, th0, om0, scheme="trapezoid")[:2]
    return Va[0], Vb[0], Vc[0], th[0], om[0]


def roll(ckpt, V, th_t, om_t, kp, ki):
    torch.set_default_dtype(torch.float32)
    m, ck = load_checkpoint(ROOT / "runs" / ckpt)
    W, S = ck["data_meta"]["W"], ck["data_meta"]["S"]
    t = ck["t_local"]; t_ext = torch.cat([t, t[-1:] + float(t[1] - t[0])])
    g = bool(getattr(m, "n_extra", 0))
    th0, om0 = th_t[0].float(), om_t[0].float()
    pth, pom = [], []
    for k in range(W):
        sl = slice(k * S, (k + 1) * S)
        p, o = predict_window(m, ck, th0, om0, V[0][sl].float(), V[1][sl].float(),
                              V[2][sl].float(), t_ext, kp if g else None, ki if g else None)
        pth.append(p[:-1]); pom.append(o[:-1]); th0, om0 = p[-1], o[-1]
    torch.set_default_dtype(torch.float64)
    return torch.cat(pth).double().numpy(), torch.cat(pom).double().numpy()


def contenders(models, W_label, fname):
    import matplotlib.pyplot as plt
    Va, Vb, Vc, th_t, om_t = truth(*THEIRS)
    t = np.arange(N) * DT
    ramp = th_t[0].item() + OMEGA_BASE * t
    fig, ax = plt.subplots(2, 2, figsize=(15, 8), sharex=True,
                           gridspec_kw=dict(height_ratios=[2, 1]))
    ax[0, 0].plot(t, (th_t.numpy() - ramp), "k", lw=2.4, label="truth (trapezoid solver)", zorder=1)
    ax[0, 1].plot(t, om_t.numpy(), "k", lw=2.4, label="truth", zorder=1)
    print(f"\n{W_label}   theta RMS / omega RMS over the whole 0.5 s rollout:")
    for lab, ck, col in models:
        pth, pom = roll(ck, (Va, Vb, Vc), th_t, om_t, *THEIRS)
        eth = pth - th_t.numpy(); eom = pom - om_t.numpy()
        print(f"   {lab:14s} theta {np.sqrt((eth**2).mean()):.3e} rad   omega {np.sqrt((eom**2).mean()):.3e} rad/s")
        ax[0, 0].plot(t, pth - ramp, col, lw=1.1, alpha=0.9, label=lab)
        ax[0, 1].plot(t, pom, col, lw=1.1, alpha=0.9, label=lab)
        ax[1, 0].semilogy(t, np.abs(eth), col, lw=1.0, label=lab)
        ax[1, 1].semilogy(t, np.abs(eom), col, lw=1.0, label=lab)
    ax[0, 0].set_ylabel(r"$\theta - (\theta_0 + \omega_{base}t)$  [rad]")
    ax[0, 1].set_ylabel(r"$\omega$  [rad/s]")
    ax[1, 0].set_ylabel(r"$|\theta$ error$|$ [rad]"); ax[1, 1].set_ylabel(r"$|\omega$ error$|$ [rad/s]")
    for a in ax.ravel():
        a.grid(alpha=0.3); a.legend(fontsize=7.5)
    for a in ax[1]:
        a.set_xlabel("time [s]")
    fig.suptitle(f"{W_label} — recurrent rollout over 0.5 s, no ground truth fed back.  "
                 f"Kp=25, Ki=300.  Top: prediction vs truth.  Bottom: absolute error.", fontsize=11)
    fig.tight_layout(); OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / fname, dpi=160); print(f"-> {OUT/fname}")


def gain_showcase():
    """The argument for tunable gains, in one figure."""
    import matplotlib.pyplot as plt
    tunings = [(12.0, 500.0, r"underdamped  $\zeta$=0.27"),
               (25.0, 300.0, r"THEIR tuning  $\zeta$=0.72"),
               (45.0, 200.0, r"overdamped  $\zeta$=1.59")]
    fig, ax = plt.subplots(2, 3, figsize=(16, 7.2), sharex=True,
                           gridspec_kw=dict(height_ratios=[2, 1]))
    for c, (kp, ki, ttl) in enumerate(tunings):
        Va, Vb, Vc, th_t, om_t = truth(kp, ki)
        t = np.arange(N) * DT
        ax[0, c].plot(t, om_t.numpy(), "k", lw=2.4, label="truth", zorder=1)
        for lab, ckpt, col in [("Kp,Ki as INPUTS", "famK_W40_n5000_W40_F4_mf503_wp0.3_s0sp0_g.pth", "tab:red"),
                               ("fixed at 25/300", "famH_W40_n5000_W40_F4_mf503_wp0.3_s0sp0.pth", "tab:blue")]:
            _, pom = roll(ckpt, (Va, Vb, Vc), th_t, om_t, kp, ki)
            e = pom - om_t.numpy()
            ax[0, c].plot(t, pom, col, lw=1.2, label=f"{lab}  ({np.sqrt((e**2).mean()):.2e})")
            ax[1, c].semilogy(t, np.abs(e), col, lw=1.0, label=lab)
        ax[0, c].set_title(f"Kp={kp:g}, Ki={ki:g}   {ttl}", fontsize=10)
        ax[0, c].legend(fontsize=8); ax[1, c].set_xlabel("time [s]")
        for r in (0, 1):
            ax[r, c].grid(alpha=0.3)
    ax[0, 0].set_ylabel(r"$\omega$ [rad/s]"); ax[1, 0].set_ylabel(r"$|\omega$ error$|$ [rad/s]")
    fig.suptitle("Why the gains have to be INPUTS: one model across three controller tunings.\n"
                 "The fixed-gain model is excellent at the tuning it was trained on and wrong everywhere else.",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "06_gain_showcase.png", dpi=160)
    print(f"-> {OUT/'06_gain_showcase.png'}")


if __name__ == "__main__":
    contenders(MODELS[:4], "W = 40   (12.5 ms windows, 80 calls/simulated second)", "04_contenders_W40.png")
    contenders(MODELS[4:], "W = 20   (25 ms windows, 40 calls/simulated second)", "05_contenders_W20.png")
    gain_showcase()
