"""Cost against the simulator step, batch 1, compiled (Numba venv, run from this folder).

When the PLL is solved in lockstep with an EMT simulator at step dt, the trapezoid solver and
the one-step NN of [1] are called 1/dt times per simulated second. The window operator is
called 80 times whatever dt is; only its output product grows with the points per window.
Waveforms are synthetic (the cost does not depend on the values); the flagship kernel is first
checked against the deployed rollout at 100 us.
"""
import json, time
import numpy as np
from kernels import solver, theirs, ours_fine
from numba_solver import run as solver_limited

F = np.load("flagship_weights.npz"); C = np.load("cases.npz"); H = np.load("head_to_head.npz")
mu, sd, kmu, ksd, imu, isd, wb = F["norm"]
bw = tuple(np.ascontiguousarray(F[f"b{k}{i}"]) for i in range(4) for k in ("W", "b"))
kpn, kin = (25 - kmu) / ksd, (300 - imu) / isd
sc, sh = H["their_scaler"].reshape(-1), H["their_shifter"].reshape(-1)
their = (np.ascontiguousarray(H["their_W1"] * sc), H["their_W1"] @ sh + H["their_B1"].reshape(-1),
         np.ascontiguousarray(H["their_W2"]), H["their_B2"].reshape(-1).copy())

def trunk_basis(t):
    x = np.column_stack([t] + [f(503.0 * k / 4 * t) for k in range(1, 5) for f in (np.sin, np.cos)])
    assert x.shape[1] == F["trunk_in"].shape[1]
    for i in range(4):
        x = x @ F[f"tW{i}"].T + F[f"tb{i}"]; x = np.tanh(x) if i < 3 else x
    return np.ascontiguousarray(x.T)

# the trunk-feature formula must reproduce the exported one exactly, and the kernel the deployed rollout
assert np.allclose(np.column_stack([F["t_ext"]] + [f(503.0 * k / 4 * F["t_ext"]) for k in range(1, 5) for f in (np.sin, np.cos)]), F["trunk_in"], atol=1e-6)   # stored in float32
chk = ours_fine(C["Va0"], C["Vb0"], C["Vc0"], 1, float(C["th0_0"]), float(C["om0_0"]), *bw,
                trunk_basis(F["t_ext"]), mu, sd, kpn, kin, wb, F["t_ext"])
print(f"flagship fine-output kernel vs deployed rollout @100 us: max|dtheta| = {np.abs(chk - F['ref_theta']).max():.1e} rad\n")

def med(f, n):
    f(); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 2e3)

res = {}
print(f"{'dt [us]':>8s} {'solver lim':>11s} {'solver unl':>11s} {'one-step NN':>12s} {'ours':>8s}   (ms per simulated second)")
for dt in (100e-6, 50e-6, 25e-6, 10e-6, 5e-6, 2e-6):
    n = int(round(0.5 / dt)); t = np.arange(n) * dt; w = 2 * np.pi * 50; r = np.random.default_rng(0)
    Va, Vb, Vc = (np.cos(w * t + p) + 0.1 * (r.random(n) - .5) for p in (0, -2 * np.pi / 3, 2 * np.pi / 3))
    m = int(round(12.5e-3 / dt)); k = m // 125; tf = np.arange(m + 1) * dt; TTf = trunk_basis(tf)
    reps = 15 if dt >= 25e-6 else 5
    row = {"solver_limited": med(lambda: solver_limited(Va, Vb, Vc, 0.3, 5.0, dt, 25.0, 300.0, 18.8496, 0.05), reps),
           "solver_unlimited": med(lambda: solver(Va, Vb, Vc, 0.3, 5.0, dt, 25.0, 300.0), reps),
           "one_step_nn": med(lambda: theirs(Va, Vb, Vc, 0.3, 5.0, dt, 300.0, *their), reps),
           "ours": med(lambda: ours_fine(Va, Vb, Vc, k, 0.3, 5.0, *bw, TTf, mu, sd, kpn, kin, wb, tf), reps)}
    res[f"{dt * 1e6:g}"] = row
    print(f"{dt * 1e6:8g} " + " ".join(f"{row[c]:11.2f}" for c in row))
json.dump(res, open("compiled_scaling.json", "w"), indent=1)
print("-> compiled_scaling.json")
