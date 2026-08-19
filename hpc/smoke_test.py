#!/usr/bin/env python
"""Five-second check that the environment can actually train, not just import.

Builds the real model, runs one real optimiser step through compute_theta_omega
(forward + two autograd.grad with create_graph + backward + SOAP), and reports the
step time. Catches broken torch builds, a missing SOAP, a mismatched numpy, and the
double-backward path all at once -- before you queue 60 jobs.

    python hpc/smoke_test.py
"""
import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pll_operator import Unstacked_DeepONet
from pll_residual import compute_theta_omega


def main():
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    threads = torch.get_num_threads()
    print(f"torch {torch.__version__} | device {dev} | {threads} threads")

    from pytorch_optimizer import SOAP

    # optional arg: sensors-per-window S. 50=W100, 125=W40, 250=W20, 500=W10
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 125
    B, F, mf = 512, 4, 503.0
    steps_per_epoch = {50: 903, 125: 361, 250: 181, 500: 90}.get(S, 361)
    torch.manual_seed(0)
    model = Unstacked_DeepONet(ov={"S_win": S, "F": F, "max_freq": mf}).to(dev)
    opt = SOAP(model.parameters(), lr=3e-3, betas=(.95, .95),
               weight_decay=.01, precondition_frequency=10)
    print(f"params {sum(p.numel() for p in model.parameters()):,} | "
          f"branch_in {model.branch_sizes[0]} trunk_in {model.trunk_sizes[0]} "
          f"output_dim {model.output_dim}")

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
        return loss.detach().item()

    for _ in range(3):
        step()
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        last = step()
    if dev == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 10 * 1000

    assert last == last, "loss is NaN -- the environment is broken"
    print(f"one optimiser step: {ms:.1f} ms   "
          f"(reference: M1 Max MPS ~26 ms with SOAP, cluster CPU 2-4 threads ~90-110 ms)")
    W = {50: 100, 125: 40, 250: 20, 500: 10}.get(S, "?")
    epoch_s = ms * steps_per_epoch / 1000
    print(f"-> W={W} (S={S}): ~{epoch_s:.0f} s/epoch at n_runs=5000, "
          f"~{epoch_s * 190 / 3600:.1f} h for a 190-epoch run")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
