"""ONE definition of "the common test". Import it; do not re-implement it.

Four scripts had grown their own copy of this: `gain_sensitivity.py`,
`exp16_report.py`, `speed_accuracy.py` and `rahul_contenders.py`. That is not a line-count
problem, it is a correctness one. Every cross-family comparison in this project rests on
"both models saw the same trajectories", and with four copies a change to the initial-
condition draw in one of them silently stops three figures from being comparable while
every one of them still runs and still prints plausible numbers. The retraction history
here is entirely this failure mode -- F48, F59, F61 -- so the harness lives in one file.

WHY THIS EXISTS AND `rollout_metrics` IN pll_infer.py DOES NOT REPLACE IT. That one scores
a model on windows drawn from a stored dataset, i.e. on its OWN family's split. This one
generates fresh trajectories at gains and an omega range the caller names, so two models
trained on different families can be put on the same axis. Per-family validation numbers
are not comparable across families; that is the whole lesson of F59/F61.

    truth(kp, ki, n_runs)                     clean batch, omega0 in +/-2
    truth(kp, ki, n_runs, kind="sag")         same ICs, one voltage sag added
    rollout(model, ck, V, th0, om0, kp, ki)   recurrent prediction over a whole run
    rollout_rms(model, ck, V, th_t, om_t)     mean per-run theta RMS, optionally timed

`rahul_contenders.py` deliberately keeps its own single hand-picked trajectory -- it draws
one legible trace rather than averaging a batch -- but uses `rollout` from here so the
recurrence and the handover are shared.
"""
import time

import numpy as np
import torch

from PLL_Simulator import PLLSimulator

DT, N = 100e-6, 5000            # 0.5 s at the step every deliverable model was trained on
OMEGA0_RANGE = 2.0              # warm co-simulation, not cold acquisition
SAG = (0.15, 0.06, 0.70)        # t0 [s], duration [s], RETAINED voltage [pu]
JUMP = (0.20, 40.0)             # t0 [s], angle [deg]


def truth(kp, ki, n_runs, kind="clean", seed=0, dt=DT, n=N, omega0=OMEGA0_RANGE):
    """Ground-truth trajectories at ONE (Kp, Ki). Returns Va, Vb, Vc, theta, omega.

    The seed is consumed in a fixed draw order, so two calls with the same seed give
    IDENTICAL initial conditions and grid waveforms regardless of kp, ki or kind. That is
    what makes a (Kp,Ki) grid and a clean/sag/jump comparison paired rather than three
    independent experiments.

    Leaves the default dtype at float64 -- the caller flips to float32 around inference.
    """
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    sim = PLLSimulator(dt=dt)
    sim.N, sim.n_runs = n, n_runs
    sim.t = (torch.arange(n) * dt).reshape(1, n)
    sim.physics.Kp = torch.full((n_runs,), float(kp), dtype=torch.float64)
    sim.physics.Ki = torch.full((n_runs,), float(ki), dtype=torch.float64)
    ga = (torch.rand(n_runs, 1) * 2 - 1) * torch.pi
    fo = (torch.rand(n_runs, 1) * 2 - 1) * 0.2
    ao = (torch.rand(n_runs, 1) * 2 - 1) * 0.05
    th0 = ga.squeeze(-1) + (torch.rand(n_runs) * 2 - 1) * 0.5 * torch.pi
    th0 = (th0 + torch.pi) % (2 * torch.pi) - torch.pi
    om0 = (torch.rand(n_runs) * 2 - 1) * omega0
    one = torch.ones(n_runs, 1, dtype=torch.float64)
    sag = (one * SAG[0], one * SAG[1], one * SAG[2]) if kind == "sag" else None
    jump = (one * JUMP[0], one * np.deg2rad(JUMP[1])) if kind == "jump" else None
    if kind not in ("clean", "sag", "jump"):
        raise ValueError(f"kind must be clean, sag or jump; got {kind!r}")
    Va, Vb, Vc = sim._grid_phases(fo, ao, ga, jump=jump, sag=sag)
    with torch.no_grad():
        th, om = sim.simulate_batch(Va, Vb, Vc, th0, om0, scheme="trapezoid")[:2]
    return Va, Vb, Vc, th, om


def load_f32(path):
    """Load a checkpoint for INFERENCE. Always use this, never bare load_checkpoint.

    `load_checkpoint` builds the network at whatever `torch.get_default_dtype()` happens
    to be. `truth()` leaves it at float64, so a bare load right after it returns a float64
    model, which then meets float32 sensor inputs and dies with

        RuntimeError: mat1 and mat2 must have the same dtype, but got Float and Double

    -- and the ordering that avoids it is invisible at the call site, so it gets broken
    every time this code is touched. Loading through here makes the order irrelevant.
    """
    torch.set_default_dtype(torch.float32)
    from train_pll import load_checkpoint          # deferred: avoids a circular import
    return load_checkpoint(path)


def rollout(model, ck, V, th0, om0, kp=None, ki=None):
    """One run, W windows, the model's own final state handed to the next window.

    `V` is (Va, Vb, Vc) for THIS run, each of length W*S. Gains are passed only to models
    that take them -- `predict_window` refuses a gains model without them, and a
    fixed-gain model rejects being given them, so the branch is decided here from the
    checkpoint rather than by every caller.
    """
    W, S = ck["data_meta"]["W"], ck["data_meta"]["S"]
    t = ck["t_local"]
    t_ext = torch.cat([t, t[-1:] + float(t[1] - t[0])])
    want = bool(getattr(model, "n_extra", 0))
    a, b = (kp, ki) if want else (None, None)
    from pll_infer import predict_window          # deferred: avoids a circular import
    th, om = [], []
    for k in range(W):
        sl = slice(k * S, (k + 1) * S)
        p, o = predict_window(model, ck, th0, om0, V[0][sl].float(), V[1][sl].float(),
                              V[2][sl].float(), t_ext, a, b)
        th.append(p[:-1]); om.append(o[:-1])
        th0, om0 = p[-1], o[-1]                   # the handover
    return torch.cat(th), torch.cat(om)


def rollout_rms(model, ck, V, th_t, om_t, kp=None, ki=None, timed=False, horizon=None):
    """Mean over runs of the per-run theta RMS. With timed=True also ms per simulated
    second, batch 1, measured on the deployed path (autograd on)."""
    torch.set_default_dtype(torch.float32)
    errs, t0 = [], time.perf_counter()
    for r in range(th_t.shape[0]):
        pth, _ = rollout(model, ck, (V[0][r], V[1][r], V[2][r]),
                         th_t[r, 0].float(), om_t[r, 0].float(), kp, ki)
        errs.append(float((pth.double() - th_t[r]).pow(2).mean().sqrt()))
    elapsed = time.perf_counter() - t0
    torch.set_default_dtype(torch.float64)
    if not timed:
        return float(np.mean(errs))
    W, S = ck["data_meta"]["W"], ck["data_meta"]["S"]
    horizon = horizon or W * S * ck["data_meta"]["dt"]
    return float(np.mean(errs)), elapsed / th_t.shape[0] / horizon * 1000
