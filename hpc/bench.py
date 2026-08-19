#!/usr/bin/env python
"""Measure the real optimiser-step cost on THIS machine, every window size, every
thread count, in ONE process. Prints a table and, if anything fails, the traceback
that caused it -- so a failed benchmark is still an informative benchmark.

    python hpc/bench.py            # auto: cuda if present, else cpu
    python hpc/bench.py cpu        # force cpu even on a GPU node
"""
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pll_operator import Unstacked_DeepONet
from pll_residual import compute_theta_omega

# S -> (W, optimiser steps per epoch at n_runs=5000, counting val at half cost)
SHAPES = {125: (40, 361), 500: (10, 90), 50: (100, 903), 250: (20, 181)}


def time_step(dev, S, threads, B=512, F=4, mf=503.0, iters=10):
    if dev == "cpu":
        torch.set_num_threads(threads)
    from pytorch_optimizer import SOAP
    torch.manual_seed(0)
    model = Unstacked_DeepONet(ov={"S_win": S, "F": F, "max_freq": mf}).to(dev)
    opt = SOAP(model.parameters(), lr=3e-3, betas=(.95, .95),
               weight_decay=.01, precondition_frequency=10)
    branch = torch.randn(B, 3 + 3 * S, device=dev)
    Vq = torch.randn(B, S, 1, device=dev)
    tth = torch.randn(B, S, 1, device=dev)
    tom = torch.randn(B, S, 1, device=dev)
    t_local = (torch.arange(S, device=dev) * 1e-4).float()

    def step():
        tq = t_local.view(1, -1, 1).expand(B, -1, 1).clone().requires_grad_(True)
        o = compute_theta_omega(model, tq, branch, Vq, omega_nominal=0.0)
        loss = (nn.functional.mse_loss(o["theta"], tth)
                + nn.functional.mse_loss(o["omega"], tom)
                + 0.3 * ((o["res_theta"] / 50).pow(2).mean()
                         + (o["res_omega"] / 600).pow(2).mean()))
        opt.zero_grad(); loss.backward(); opt.step()

    for _ in range(3):
        step()
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    if dev == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    want = sys.argv[1] if len(sys.argv) > 1 else None
    dev = want or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__}   device {dev}")
    if dev == "cuda":
        print(f"gpu   {torch.cuda.get_device_name(0)}")
    print(f"cores visible to LSF: {os.environ.get('LSB_DJOB_NUMPROC', '?')}   "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}")
    print()

    thread_list = [1, 2, 4, 8] if dev == "cpu" else [0]      # 0 = n/a on GPU
    print(f"{'W':>4s} {'S':>5s} {'thr':>4s} {'ms/step':>9s} {'s/epoch':>9s} {'h/run(190ep)':>13s}")
    print("-" * 52)
    results = {}
    for S, (W, spe) in SHAPES.items():
        for thr in thread_list:
            try:
                ms = time_step(dev, S, thr or 1)
                epoch_s = ms * spe / 1000
                hours = epoch_s * 190 / 3600
                results[(W, thr)] = hours
                print(f"{W:4d} {S:5d} {thr if thr else '-':>4} "
                      f"{ms:9.1f} {epoch_s:9.1f} {hours:13.2f}")
            except Exception:
                print(f"{W:4d} {S:5d} {thr if thr else '-':>4}   FAILED")
                traceback.print_exc(file=sys.stdout)
                print()

    if results:
        print()
        print("exp1 = 48 jobs at W=40 ; exp2 = 12 jobs at W=10/20/100")
        for thr in thread_list:
            w40 = results.get((40, thr))
            longest = max((results.get((W, thr), 0) for W in (10, 20, 100)), default=0)
            if w40:
                print(f"  threads={thr if thr else '-'}: W=40 run {w40:.1f} h, "
                      f"longest exp2 run {longest:.1f} h  "
                      f"-> set walltime >= {max(w40, longest) * 1.6:.0f} h")


if __name__ == "__main__":
    main()
