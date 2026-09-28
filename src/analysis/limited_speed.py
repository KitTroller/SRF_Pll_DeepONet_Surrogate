"""Speed on the LIMITED system: the flagship against the limited trapezoid solver.

    python src/analysis/limited_speed.py runs/famO40k_W40_n40000_W40_F4_mf503_wp0.3_s6sp0_L3_w128_ao_g.pth

graphs/12's 41x is the UNLIMITED solver against an unlimited model. The paper's system has
the frequency limiter, whose solver has to iterate theta AND omega together in its Newton
loop (PLL_Simulator._integrator_step_limited), so it is slower per step than the unlimited
scalar Newton. The network's cost does not depend on the physics at all. The notes quote
98x for this architecture, measured on a different machine; this re-measures it here.

Everything is BATCH 1, per trajectory (F41: an EMT simulator integrates one trajectory,
and the solver's advantage under batching is a different claim). The solver is timed on a
single trajectory, not on a batch divided by horizon. Every timing is the median of
`--repeats` after one warm-up, in one process, on the same CPU, one after another.
"""

# src/ on the path: these scripts live in src/analysis/ but import the pipeline
# modules (paths, PLL_Simulator, train_pll, sweep) that stay in src/. Running
# `python src/analysis/foo.py` puts src/analysis on sys.path, not src/.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import argparse
import platform
import statistics as st

import torch

import PLL_Simulator as PS
from common_test import load_f32
from paths import ROOT
from speed_benchmark import build_case, solve_at, deeponet_at


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--n_nn", type=int, default=8, help="trajectories for the surrogate timing "
                   "(run one after another, so per trajectory is still batch 1)")
    a = p.parse_args()

    model, ck = load_f32(ROOT / a.ckpt if not a.ckpt.startswith("/") else a.ckpt)
    m = ck["data_meta"]; W, S, dt = m["W"], m["S"], m["dt"]
    kp, ki = float(PS.pll_constants.Pll.Kp), float(PS.pll_constants.Pll.Ki)
    print(f"{platform.processor() or platform.machine()} | torch {torch.__version__} | "
          f"{torch.get_num_threads()} threads | dt={dt*1e6:.0f} us, 0.5 s per trajectory\n")

    def med(fn):
        fn()                                            # warm-up
        return st.median(fn() for _ in range(a.repeats))

    rows = {}
    for label, lim in [("solver, UNLIMITED", None), ("solver, LIMITED", m["freq_limit"])]:
        PS.pll_constants.freq_limit = lim               # read by every new PLLSimulator
        case = build_case(dt, 0.5, 1)                   # ONE trajectory, solved at its own dt
        rows[label] = med(lambda: solve_at(case, dt, timeit=True)[1])
    case = build_case(dt, 0.5, a.n_nn)
    rows["flagship DeepONet"] = med(lambda: deeponet_at(case, model, ck, W, S, dt, timeit=True,
                                                        kp=kp, ki=ki)[1])

    nn = rows["flagship DeepONet"]
    for k, v in rows.items():
        print(f"  {k:22s} {v:9.1f} ms per simulated second" + ("" if k.startswith("flagship")
              else f"   -> surrogate is {v / nn:5.1f}x faster"))
    print(f"\n  limited / unlimited solver cost: {rows['solver, LIMITED'] / rows['solver, UNLIMITED']:.2f}x")


if __name__ == "__main__":
    main()
