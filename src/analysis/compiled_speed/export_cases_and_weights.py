"""Step 1 of the compiled benchmark (F82). Run in the PROJECT venv from the repo root:

    .venv/bin/python src/analysis/compiled_speed/export_cases_and_weights.py

Writes, next to this file:
  cases.npz             three LIMITED test cases (100 us, 0.5 s, omega0 +/-20) with the project
                        solver's theta: the ground truth the Numba solver must reproduce
  flagship_weights.npz  the flagship's branch/trunk weights, the trunk input on t_ext, the
                        normalisation, and the deployed predict_window rollout on case 0
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent)); sys.path.insert(0, str(HERE.parent))

import numpy as np
import torch

import PLL_Simulator as PS
from common_test import load_f32
from pll_infer import predict_window
from pll_residual import build_trunk_input
from speed_benchmark import build_case, solve_at
from train_pll import OMEGA_BASE

FLAG = "runs/famO40k_W40_n40000_W40_F4_mf503_wp0.3_s6sp0_L3_w128_ao_g.pth"

PS.pll_constants.freq_limit = 18.8496
cases = {}
for s in range(3):
    case = build_case(100e-6, 0.5, 1, seed=s)
    th, _ = solve_at(case, 100e-6)
    for k in ("Va", "Vb", "Vc"):
        cases[f"{k}{s}"] = case[k][0].numpy()
    cases[f"th0_{s}"], cases[f"om0_{s}"] = float(case["theta0"][0]), float(case["omega0"][0])
    cases[f"th{s}"] = th[0].numpy()
np.savez(HERE / "cases.npz", **cases)

model, ck = load_f32(FLAG)
lin = lambda seq: [m for m in seq if isinstance(m, torch.nn.Linear)]
W = {}
for name, net in [("b", model.branch_net.model), ("t", model.trunk_net.model)]:
    for i, layer in enumerate(lin(net)):
        W[f"{name}W{i}"] = layer.weight.detach().numpy().astype(np.float64)
        W[f"{name}b{i}"] = layer.bias.detach().numpy().astype(np.float64)
t = ck["t_local"]; t_ext = torch.cat([t, t[-1:] + (t[1] - t[0])])
W["trunk_in"] = build_trunk_input(t_ext.view(1, -1, 1), model.F, model.max_freq)[0].numpy().astype(np.float64)
(kmu, ksd), (imu, isd) = ck["gstat"]
W["norm"] = np.array([float(ck["mu"]), float(ck["sd"]), kmu, ksd, imu, isd, OMEGA_BASE])
W["t_ext"] = t_ext.numpy().astype(np.float64)

Va, Vb, Vc = (torch.tensor(cases[k + "0"], dtype=torch.float32) for k in ("Va", "Vb", "Vc"))
th0, om0, ths = torch.tensor(cases["th0_0"]), torch.tensor(cases["om0_0"]), []
with torch.no_grad():
    for w in range(40):
        sl = slice(w * 125, (w + 1) * 125)
        th, om = predict_window(model, ck, th0, om0, Va[sl], Vb[sl], Vc[sl], t_ext, 25.0, 300.0)
        ths.append(th[:-1].numpy()); th0, om0 = th[-1], om[-1]
W["ref_theta"] = np.concatenate(ths).astype(np.float64)
np.savez(HERE / "flagship_weights.npz", **W)
print(f"-> {HERE / 'cases.npz'} and {HERE / 'flagship_weights.npz'}")
