"""Compiled Fig. 2, step 1. Run in the PROJECT venv from the repo root:

    .venv/bin/python src/analysis/compiled_speed/export_head_to_head.py

graphs/12's head-to-head is UNLIMITED physics, fixed Kp=25 Ki=300, inside the one-step NN's
trained range. Its accuracy numbers stay as they are (a compiled re-implementation computes
the same outputs); only the COMPUTE axis is re-measured. This saves, for one run of that case,
the reference outputs of every method as the project computes them, plus the weights, so
bench_head_to_head.py can check each compiled version reproduces its original before timing it:

  solver @100 us and @50 us (PyTorch trapezoid)       their NN @50 us and @100 us (their NumPy code)
  ours: famR L3_w128 s0 (predict_window, no anchor)   plain MLP: famB Single_PINN s0
"""
import os
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent)); sys.path.insert(0, str(HERE.parent))

import numpy as np
import torch

from common_test import load_f32
from paths import PAPER_REPO
from speed_benchmark import build_case, solve_at, deeponet_at, paper_nn_at

OURS = "runs/famR_W40_n5000_W40_F4_mf503_wp0.3_s0sp0_L3_w128.pth"
MLP = "runs/famB_W40_n5000_W40_F4_mf503_wp0.3_s0sp0_pinn.pth"
out = {}

case = build_case(12.5e-6, 0.5, 1, seed=0, their_range=True)       # graph 12's envelope, one run
for dt in (100e-6, 50e-6):
    k = int(round(dt / case["dt_fine"]))
    tag = f"{int(dt * 1e6)}"
    for v in ("Va", "Vb", "Vc"):
        out[f"{v}_{tag}"] = case[v][0, ::k].numpy()
    out[f"solver_{tag}"] = solve_at(case, dt)[0][0].numpy()
    out[f"theirs_{tag}"] = paper_nn_at(case, dt)[0][0].numpy()
out["th0"], out["om0"] = float(case["theta0"][0]), float(case["omega0"][0])

lin = lambda seq: [m for m in seq if isinstance(m, torch.nn.Linear)]
for name, path in (("ours", OURS), ("mlp", MLP)):
    torch.set_default_dtype(torch.float32)
    model, ck = load_f32(path)
    th, _ = deeponet_at(case, model, ck, 40, 125, 100e-6)
    out[f"{name}_ref"] = th[0].double().numpy()
    nets = ([("b", model.branch_net.model), ("t", model.trunk_net.model)] if name == "ours"
            else [("p", model.pinn_mlp.model)])
    for pre, net in nets:
        for i, layer in enumerate(lin(net)):
            out[f"{name}_{pre}W{i}"] = layer.weight.detach().double().numpy()
            out[f"{name}_{pre}b{i}"] = layer.bias.detach().double().numpy()
    out[f"{name}_mu"], out[f"{name}_sd"] = float(ck["mu"]), float(ck["sd"])
    t = ck["t_local"]; t_ext = torch.cat([t, t[-1:] + (t[1] - t[0])])
    from pll_residual import build_trunk_input
    out[f"{name}_trunk_in"] = build_trunk_input(t_ext.view(1, -1, 1), model.F, model.max_freq)[0].double().numpy()
    out["t_ext"] = t_ext.double().numpy()

p = np.load(os.path.join(PAPER_REPO, "NN_models", "version1.npz"))
for k in p.files:
    out[f"their_{k}"] = p[k]
from PLL_Simulator import PhysicsEquations
out["Ki"] = float(PhysicsEquations().Ki)
np.savez(HERE / "head_to_head.npz", **out)
print(f"-> {HERE / 'head_to_head.npz'}")
