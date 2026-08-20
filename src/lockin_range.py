"""Is +/-80 rad/s outside what the SRF-PLL itself can acquire in 0.5 s?

    python src/lockin_range.py

F42 found `omega_0` to be the one hard edge of the operator's envelope. Before blaming
the network, ask what the REFERENCE SOLVER does there -- if the loop itself cannot
acquire lock in the window we train on, then past some `omega_0` we are asking the
surrogate to reproduce a trajectory that has no settled behaviour to learn.

Classical PLL theory gives three ranges, and only the first is fast:
    hold-in    already locked, stays locked. Infinite for an ideal PI loop filter.
    lock-in    acquires WITHOUT cycle slips, within ~1/(zeta*wn).   dw_L ~= 2*zeta*wn
    pull-in    eventually acquires, but slips cycles on the way. Slow, and the slip
               transient is exactly what a 0.5 s window cannot contain.
With Kp=25, Ki=300: wn = sqrt(Ki) = 17.32 rad/s, zeta = Kp/(2*sqrt(Ki)) = 0.722, so
dw_L ~= 25 rad/s. The trained range is +/-20 -- just inside. That is testable, not a
citation, so this script measures it.

Isolates omega_0 on purpose: nominal grid frequency, no amplitude offset, and the PLL
starts phase-ALIGNED (theta_pll0 = theta_grid0), so the only error is in frequency.
"""
import numpy as np
import torch

from PLL_Simulator import PLLSimulator, PhysicsEquations
from paths import GRAPHS

OMEGA0_GRID = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0, 80.0]
SETTLE_TOL = 0.10          # rad. Sensor noise alone puts ~0.06 rad of jitter on eps.


def run(horizon=1.0, dt=12.5e-6, seed=0):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    ph = PhysicsEquations()
    n = len(OMEGA0_GRID)
    N = int(round(horizon / dt))

    sim = PLLSimulator(dt=dt)
    sim.N, sim.n_runs = N, n
    sim.t = (torch.arange(N) * dt).reshape(1, N)

    zeros = torch.zeros(n, 1)
    Va, Vb, Vc = sim._grid_phases(zeros, zeros, zeros)      # nominal grid, phase 0
    theta0 = torch.zeros(n)                                  # phase-ALIGNED at t=0
    omega0 = torch.tensor(OMEGA0_GRID, dtype=torch.float64)

    with torch.no_grad():
        theta, omega, _, _, _, _ = sim.simulate_batch(Va, Vb, Vc, theta0, omega0,
                                                      scheme="trapezoid")
    t = (torch.arange(N) * dt).numpy()
    theta_grid = ph.omega_0 * t                              # grid angle, init phase 0
    eps = theta.numpy() - theta_grid[None, :]                # phase error, unwrapped
    return t, eps, omega.numpy()


def analyse(t, eps, omega):
    wn = np.sqrt(PhysicsEquations().Ki)
    zeta = PhysicsEquations().Kp / (2 * wn)
    print(f"wn = {wn:.2f} rad/s   zeta = {zeta:.3f}   "
          f"predicted lock-in range 2*zeta*wn = {2*zeta*wn:.1f} rad/s")
    print(f"linear 2% settling time 4/(zeta*wn) = {4/(zeta*wn):.3f} s  "
          f"(training window is 0.5 s)\n")
    # Settling must be judged on the WRAPPED error. A PLL that slips k cycles is
    # physically locked -- Vd = +1, Vq = 0, it tracks the grid perfectly -- but its
    # UNWRAPPED angle sits k*2pi away forever. Judging on the raw error calls that
    # "never settled", which is wrong and hides the actual failure mode.
    print(f"  {'omega_0':>8s} {'Hz err':>7s} {'slips':>6s} {'locked by':>10s} "
          f"{'|eps| wrapped @0.5s':>20s} {'raw |eps| @0.5s':>16s}")
    rows = []
    for i, w0 in enumerate(OMEGA0_GRID):
        e = eps[i]
        ew = (e + np.pi) % (2 * np.pi) - np.pi
        slips = int(round(e[-1] / (2 * np.pi)))
        bad = np.where(np.abs(ew) >= SETTLE_TOL)[0]
        t_lock = t[bad[-1] + 1] if len(bad) and bad[-1] + 1 < len(t) else (0.0 if not len(bad) else np.inf)
        half = len(t) // 2
        e05w = np.abs(ew[half - 100:half]).mean()
        e05 = np.abs(e[half - 100:half]).mean()
        s = f"{t_lock:.3f} s" if np.isfinite(t_lock) else "NEVER"
        print(f"  {w0:8.1f} {w0/(2*np.pi):7.2f} {slips:6d} {s:>10s} {e05w:20.3e} {e05:16.3e}")
        rows.append((w0, slips, t_lock, e05w, e05))
    print(f"\n  slips = whole turns the unwrapped angle ends up away from the grid.")
    print(f"  A slipped run is LOCKED (Vq -> 0) but permanently k*2pi off in absolute angle,")
    print(f"  which is what our theta RMS -- an unwrapped, absolute metric -- charges it for.")
    return rows


def figure(t, eps, rows):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for i, w0 in enumerate(OMEGA0_GRID):
        trained = abs(w0) <= 20.0
        ax[0].plot(t, eps[i], lw=1.1 if trained else 1.6,
                   ls="-" if trained else "--",
                   label=f"$\\omega_0$={w0:g}" + ("" if trained else "  (outside trained)"))
    ax[0].axhline(np.pi, color="k", lw=0.8, ls=":")
    ax[0].axhline(-np.pi, color="k", lw=0.8, ls=":")
    ax[0].axvline(0.5, color="tab:red", lw=1.2)
    ax[0].annotate("end of the 0.5 s\ntraining window", (0.5, 0), xytext=(0.56, 4),
                   fontsize=8, color="tab:red",
                   arrowprops=dict(arrowstyle="->", color="tab:red", lw=1))
    ax[0].set_xlabel("time [s]"); ax[0].set_ylabel(r"phase error $\epsilon$ [rad]")
    ax[0].set_title(r"Acquisition from a pure frequency error. $\pm\pi$ = one cycle slip")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=7, ncol=2)

    w = [r[0] for r in rows]
    ax[1].plot(w, [r[1] for r in rows], "o-", color="tab:red", label="cycle slips")
    ax[1].plot(w, [0 if not np.isfinite(r[2]) else r[2] for r in rows], "s-",
               color="tab:blue", label="time to lock [s]  (wrapped error)")
    wn = np.sqrt(PhysicsEquations().Ki)
    zeta = PhysicsEquations().Kp / (2 * wn)
    ax[1].axvline(2 * zeta * wn, color="k", ls="--", lw=1.3,
                  label=f"lock-in range $2\\zeta\\omega_n$ = {2*zeta*wn:.0f}")
    ax[1].axvspan(0, 20, color="tab:green", alpha=0.12, label="trained range")
    ax[1].axhline(0.5, color="tab:red", ls=":", lw=1.1, label="0.5 s window")
    ax[1].set_xlabel(r"$\omega_0$ [rad/s]"); ax[1].set_ylabel("slips  /  seconds")
    ax[1].set_title("Where acquisition stops fitting in the window")
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=7)

    fig.suptitle("The SRF-PLL's own acquisition limit — reference solver at 12.5 us, "
                 "no network involved", fontsize=10)
    fig.tight_layout()
    GRAPHS.mkdir(exist_ok=True)
    out = GRAPHS / "16_lockin_range.png"
    fig.savefig(out, dpi=160)
    print(f"\n-> {out}")


if __name__ == "__main__":
    t, eps, omega = run()
    rows = analyse(t, eps, omega)
    figure(t, eps, rows)
