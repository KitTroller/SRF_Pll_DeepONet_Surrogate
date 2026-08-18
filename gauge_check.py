# gauge_check.py -- the physics loss is invariant under theta -> theta + a*t + b,
# omega -> omega + a.  The data loss is not.  Those two parameters are (omega0, theta0).
import torch, torch.nn as nn
from train_pll import load_checkpoint, prepare, group_split, build_branch, batches, KP, KI
from pll_residual import compute_theta_omega
from dataset_generator import Dataset_Creator

model, ck = load_checkpoint("runs/W40_F0_wp0.3_s1.pth")     # any trained checkpoint
data, meta = Dataset_Creator.load_dataset("pll_dataset.npz")
prep = prepare(data)
tr, va = group_split(prep["run_id"], 0.15, 0)
branch = build_branch(prep, ck["mu"], ck["sd"])
br, Vq, tth, tom = next(batches(
    tuple(t[va] for t in (branch, prep["Vq"], prep["target_theta"], prep["target_omega"])),
    int(va.sum()), 256, False, "cpu", False))

B = br.shape[0]
tq = prep["t_local"].view(1, -1, 1).expand(B, -1, 1).clone().requires_grad_(True)
out = compute_theta_omega(model, tq, br, Vq.unsqueeze(-1), omega_nominal=0.0)
s1, s2 = ck["s1"], ck["s2"]

print(f"{'a':>6s} {'b':>6s} | {'r1':>12s} {'r2':>12s} | {'l_th':>12s} {'l_om':>12s}")
for a, b in [(0.0, 0.0), (0.0, 0.5), (2.0, 0.0), (2.0, 0.5), (0.0, 5.0)]:
    th = out["theta"] + a * tq + b          # the zero mode
    om = out["omega"] + a
    dth = torch.autograd.grad(th, tq, torch.ones_like(th), create_graph=True, retain_graph=True)[0]
    dom = torch.autograd.grad(om, tq, torch.ones_like(om), allow_unused=True, retain_graph=True)[0]
    dom = torch.zeros_like(dth) if dom is None else dom
    r1 = ((dth - om - KP * Vq.unsqueeze(-1)) / s1).pow(2).mean()
    r2 = ((dom + torch.autograd.grad(out["omega"], tq, torch.ones_like(om),
                                     retain_graph=True)[0] * 0 - KI * Vq.unsqueeze(-1)) / s2).pow(2).mean()
    l_th = nn.functional.mse_loss(th, tth.unsqueeze(-1))
    l_om = nn.functional.mse_loss(om, tom.unsqueeze(-1))
    print(f"{a:6.1f} {b:6.1f} | {r1.item():12.6e} {r2.item():12.6e} | "
          f"{l_th.item():12.6e} {l_om.item():12.6e}")