"""Compiled Fig. 2, step 2. Run in the NUMBA venv (see README) from this folder:

    ../../../nvenv-or-yours/bin/python bench_head_to_head.py

Every method of graphs/12 re-implemented in Numba, batch 1, and CHECKED against the project's
own output (head_to_head.npz) before it is timed:
  trapezoid solver, unlimited, @100 us and @50 us   (scalar Newton, as _integrator_step_trapezoid)
  their one-step NN @50 us and @100 us               (their 5-56-2 net, exactly their inference code)
  ours, famR L3_w128                                  (lean: trunk evaluated once)
  plain MLP, famB Single_PINN                         (lean: the t-part of layer 1 evaluated once --
                                                       the same trick, so the comparison stays fair)
Writes compiled_head_to_head.json for paper_figures.py.
"""
import json
import time

import numpy as np
from numba import njit

D = np.load("head_to_head.npz")
from kernels import W0, solver, theirs, ours, mlp


def unwrap_anchor(th, th0):
    th = np.unwrap(th)
    return th + th0 - th[0]


# ---- assemble inputs
th0, om0, Ki, t_ext = float(D["th0"]), float(D["om0"]), float(D["Ki"]), D["t_ext"]
V = lambda tag: tuple(np.ascontiguousarray(D[f"{v}_{tag}"]) for v in ("Va", "Vb", "Vc"))
x = D["ours_trunk_in"]
for i in range(4):
    x = x @ D[f"ours_tW{i}"].T + D[f"ours_tb{i}"]; x = np.tanh(x) if i < 3 else x
ours_args = tuple(np.ascontiguousarray(D[f"ours_b{k}{i}"]) for i in range(4) for k in ("W", "b")) + (
    np.ascontiguousarray(x.T), float(D["ours_mu"]), float(D["ours_sd"]), t_ext)
W1 = D["mlp_pW0"]
mlp_args = (np.ascontiguousarray(W1[:, :378]), np.ascontiguousarray(D["mlp_trunk_in"] @ W1[:, 378:].T + D["mlp_pb0"]),
            np.ascontiguousarray(D["mlp_pW1"].T), D["mlp_pb1"].copy(), np.ascontiguousarray(D["mlp_pW2"].T),
            D["mlp_pb2"].copy(), float(D["mlp_mu"]), float(D["mlp_sd"]), t_ext)
sc, sh = D["their_scaler"].reshape(-1), D["their_shifter"].reshape(-1)
their_args = (np.ascontiguousarray(D["their_W1"] * sc), D["their_W1"] @ sh + D["their_B1"].reshape(-1),
              np.ascontiguousarray(D["their_W2"]), D["their_B2"].reshape(-1).copy())

runs = {
    "solver_100": (lambda: solver(*V("100"), th0, om0, 100e-6, 25.0, 300.0), D["solver_100"], None),
    "solver_50": (lambda: solver(*V("50"), th0, om0, 50e-6, 25.0, 300.0), D["solver_50"], None),
    "theirs_50": (lambda: theirs(*V("50"), th0, om0, 50e-6, Ki, *their_args), D["theirs_50"], "wrap"),
    "theirs_100": (lambda: theirs(*V("100"), th0, om0, 100e-6, Ki, *their_args), D["theirs_100"], "wrap"),
    "ours": (lambda: ours(*V("100"), th0, om0, *ours_args), D["ours_ref"], None),
    "mlp": (lambda: mlp(*V("100"), th0, om0, *mlp_args), D["mlp_ref"], None),
}
res = {}
print("compiled vs original output, then batch-1 cost (median of 30, 0.5 s runs):")
for name, (fn, ref, mode) in runs.items():
    th = fn()
    th = unwrap_anchor(th, th0) if mode == "wrap" else th
    err = float(np.abs(th[:len(ref)] - ref).max())
    fn(); ts = []
    for _ in range(30):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    res[name] = {"ms_per_sim_s": float(np.median(ts) * 2e3), "max_abs_diff_vs_original": err}
    print(f"  {name:11s} max|diff| {err:.1e} rad   {res[name]['ms_per_sim_s']:7.2f} ms per simulated second")
json.dump(res, open("compiled_head_to_head.json", "w"), indent=1)
print("-> compiled_head_to_head.json")
