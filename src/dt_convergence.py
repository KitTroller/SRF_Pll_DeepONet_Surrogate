"""Does finer sampling actually buy accuracy? Answered WITHOUT training anything.

    python src/dt_convergence.py

F48 measured that dt 100 -> 50 us improved the deployed error 1.58x, and recommended
moving to 10000 sensors. **This script shows that gain is an artefact of the noise
model, not a real improvement.** Three arms, all against a noiseless 6.25 us reference:

  noise OFF          pure integration error. The trapezoid is 2nd order, so this should
                     divide by 4 per halving -- and it does. It is also FIVE ORDERS OF
                     MAGNITUDE below the other two arms, i.e. truncation is irrelevant
                     here and everything we have been calling "discretisation error" is
                     really sensor noise.
  sigma CONST        what `_grid_phases` does today: one fixed-amplitude uniform draw per
                     sample, whatever dt is. Halving dt therefore HALVES the in-band noise
                     power, and the error falls ~1.4x per halving (= sqrt(2), the
                     signature of averaging white noise, not of a 2nd-order method).
  PSD CONST          the physical case. A real sensor's noise spectral density does not
                     change because you sample it faster; holding it fixed means
                     sigma ~ 1/sqrt(dt). Error is FLAT.

All three arms share ONE noiseless waveform generated on the finest grid and subsampled,
with noise added afterwards at each arm's sigma. Signal fixed, noise varied -- see the
comment in `_clean()` for the version of this that got it wrong and what it cost.
"""
import numpy as np
import torch

from PLL_Simulator import PLLSimulator
from paths import GRAPHS

DTS = [200e-6, 100e-6, 50e-6, 25e-6]      # 2500 / 5000 / 10000 / 20000 sensors per 0.5 s
DT_FINE = 6.25e-6
DT_REF = 100e-6                            # the dt at which noise_amplitude is calibrated


def _clean(n_runs=24, horizon=0.5, seed=0, jump=None):
    """ONE noiseless waveform on the finest grid, plus the ICs. Everything subsamples this.

    Why not regenerate per dt: `_grid_phases` draws its `torch.rand(3, n_runs, N)` noise
    block BEFORE the harmonics, and that block's size depends on N -- so regenerating at a
    different dt silently shifts the RNG and gives DIFFERENT harmonic amplitudes and decay
    constants. A first version of this script did that and the noise-off arm came out flat
    at 5e-5, which was not truncation error at all but the gap between two different
    harmonic realisations. Signal fixed, noise varied: that is the only way to read this.
    """
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    n = int(round(horizon / DT_FINE))
    s = PLLSimulator(dt=DT_FINE)
    s.N, s.n_runs = n, n_runs
    s.t = (torch.arange(n) * DT_FINE).reshape(1, n)
    s.physics.noise_amplitude = 0.0
    grid_ang = (torch.rand(n_runs, 1) * 2 - 1) * torch.pi
    freq_off = (torch.rand(n_runs, 1) * 2 - 1) * 0.2
    amp_off = (torch.rand(n_runs, 1) * 2 - 1) * 0.05
    theta0 = grid_ang.squeeze(-1) + (torch.rand(n_runs) * 2 - 1) * 0.5 * torch.pi
    theta0 = (theta0 + torch.pi) % (2 * torch.pi) - torch.pi
    omega0 = (torch.rand(n_runs) * 2 - 1) * 20.0
    jmp = None
    if jump is not None:
        # a phase step at a time deliberately OFF every coarse grid: 0.2003 s is not a
        # multiple of 200/100/50/25 us, so each dt quantises the step edge differently.
        # That quantisation is the effect being measured.
        t0, delta = jump
        jmp = (torch.full((n_runs, 1), t0), torch.full((n_runs, 1), delta))
    V = s._grid_phases(freq_off, amp_off, grid_ang, jump=jmp)
    return V, theta0, omega0, n_runs


def run(dt, noise, clean, horizon=0.5, seed=1):
    """Subsample the fixed clean waveform onto this dt, then add noise at THIS sigma."""
    V, theta0, omega0, n_runs = clean
    k = int(round(dt / DT_FINE))
    n = int(round(horizon / dt))
    torch.manual_seed(seed)                              # noise only
    Va, Vb, Vc = (v[:, ::k][:, :n] for v in V)
    if noise:
        e = noise * (torch.rand(3, n_runs, n) - 0.5)     # v_nominal = 1 pu
        Va, Vb, Vc = Va + e[0], Vb + e[1], Vc + e[2]
    s = PLLSimulator(dt=dt)
    s.N, s.n_runs = n, n_runs
    s.t = (torch.arange(n) * dt).reshape(1, n)
    with torch.no_grad():
        return s.simulate_batch(Va, Vb, Vc, theta0, omega0, scheme="trapezoid")[0]


ARMS = {
    "noise OFF  (pure integration error)":      (lambda dt: 0.0,                        "^-.", "tab:green"),
    "sigma CONST  (as `_grid_phases` codes it)": (lambda dt: 0.1,                        "o-",  "tab:blue"),
    "PSD CONST  (physically consistent)":       (lambda dt: 0.1 * (DT_REF / dt) ** 0.5, "s--", "tab:red"),
}


def sweep(ax, clean, arms, title):
    ref = run(DT_FINE, 0.0, clean)                            # noiseless truth, finest grid
    for label, (sigma_of, fmt, col) in arms.items():
        print(f"\n{label}")
        print(f"  {'dt [us]':>8s} {'sensors':>8s} {'sigma':>7s} {'theta RMS':>12s} {'gain':>8s}")
        xs, ys, prev = [], [], None
        for dt in DTS:
            k = int(round(dt / DT_FINE))
            n = int(round(0.5 / dt))
            sig = sigma_of(dt)
            e = float((run(dt, sig, clean) - ref[:, ::k][:, :n]).pow(2).mean().sqrt())
            g = f"{prev / e:7.2f}x" if prev else f"{'-':>8s}"
            print(f"  {dt*1e6:8.1f} {n:8d} {sig:7.3f} {e:12.3e} {g}")
            xs.append(dt * 1e6); ys.append(e); prev = e
        ax.loglog(xs, ys, fmt, color=col, lw=1.7, ms=6, label=label)
    ax.set_xlabel(r"timestep [$\mu$s]   2500 / 5000 / 10000 / 20000 sensors per 0.5 s")
    ax.set_ylabel(r"$\theta$ RMS vs a noiseless 6.25 $\mu$s reference [rad]")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7.5, loc="best")


def main():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.4), sharey=True)

    print("\n=================== SMOOTH input (no fault) ===================")
    sweep(ax[0], _clean(), ARMS,
          "SMOOTH input: finer sampling buys nothing at fixed noise PSD\n"
          "(the blue gain is the noise model shrinking, not the integrator)")

    print("\n=================== STEP input (60 deg phase jump) ===================")
    # only the physically-consistent arm matters here -- the question is whether a REAL
    # gain survives once the noise artefact is removed.
    step_arms = {k: v for k, v in ARMS.items() if "PSD CONST" in k or "noise OFF" in k}
    sweep(ax[1], _clean(jump=(0.2003, np.deg2rad(60.0))), step_arms,
          "STEP input: a 60 deg phase jump at t = 0.2003 s\n"
          "a discontinuity is not bandlimited, so dt quantises the step edge")

    fig.suptitle("Does finer sampling buy accuracy? Solver only — no network anywhere",
                 fontsize=11)
    fig.tight_layout()
    GRAPHS.mkdir(exist_ok=True)
    fig.savefig(GRAPHS / "19_dt_convergence.png", dpi=160)
    print(f"\n-> {GRAPHS / '19_dt_convergence.png'}")


if __name__ == "__main__":
    main()
