"""Checks for the omega anchor (docs/ANCHOR_GUIDE.html). Run it after each step.

    python src/analysis/check_anchor.py            # the fast checks, ~1 min
    python src/analysis/check_anchor.py --train    # + a 2-epoch training smoke test

Every check prints PASS or FAIL with the reason. Nothing here writes into runs/ or
Hyperparameter_sweep/ -- the training smoke test uses a temporary folder.

What the anchor must guarantee, and what each check proves:
  1  old checkpoints still load, as anchor_omega=False        (nothing already trained changes)
  2  old checkpoints score EXACTLY as before when not anchored (the default path is untouched)
  3  post-hoc anchoring reproduces F73's number               (inference anchor is right)
  4  omega(0) == omega0 exactly, in BOTH paths                (the anchor itself)
  5  training path == inference path on the same inputs      (the bug class that bit exp28 twice)
  6  domega/dt is unchanged by the anchor                    (the physics residual is untouched)
  7  gradients reach every parameter                         (N(0) is part of the model, not detached)
  8  save -> load keeps the flag and the outputs              (a 30-hour run does not come back wrong)
  9  --anchor_omega exists on the CLI and in train_pll.main
 10  (--train) 2 epochs run end to end; tag ends _ao_g; record has anchor_omega
"""

# src/ on the path: these scripts live in src/analysis/ but import the pipeline
# modules (paths, PLL_Simulator, train_pll, sweep) that stay in src/. Running
# `python src/analysis/foo.py` puts src/analysis on sys.path, not src/.
# Same pattern as hpc/generate_family.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import argparse
import inspect
import json
import subprocess
import tempfile

import torch

import train_pll as T
from dataset_generator import Dataset_Creator
from paths import ROOT
from pll_infer import predict_window, _gains
from pll_operator import Unstacked_DeepONet
from pll_residual import compute_theta_omega

CKPT = "famO_W40_n5000_W40_F4_mf503_wp0.3_s0sp0_L3_w128_g.pth"   # the deliverable, seed 0
F73_NORMAL, F73_ANCHOR = 7.637e-4, 5.876e-4                      # handover_test.py, 150 runs
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   -- {detail}" if detail else ""))


def rollout_rms(model, ck, prep, runs, W, anchor):
    t = ck["t_local"]; t_ext = torch.cat([t, t[-1:] + (t[1] - t[0])])
    errs = []
    for r in runs:
        row0 = r * W
        th0, om0, pth = prep["theta0_abs"][row0], prep["omega0"][row0], []
        for k in range(W):
            kp, ki = _gains(prep, row0 + k)
            th, om = predict_window(model, ck, th0, om0, prep["Va"][row0 + k], prep["Vb"][row0 + k],
                                    prep["Vc"][row0 + k], t_ext, kp, ki, anchor_omega=anchor)
            pth.append(th[:-1]); th0, om0 = th[-1], om[-1]
        e = torch.cat(pth) - torch.cat([prep["theta_abs"][row0 + k] for k in range(W)])
        errs.append(e.pow(2).mean().sqrt())
    return float(torch.stack(errs).mean())


def fresh(anchor, like=None):
    torch.manual_seed(0)
    m = Unstacked_DeepONet(ov={"S_win": 125, "n_extra": 2, "n_layers": 3, "width": 128,
                               "F": 4, "max_freq": 503.0, "anchor_omega": anchor})
    if like is not None:
        m.load_state_dict(like.state_dict())
    return m.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true", help="also run the 2-epoch training smoke test")
    a = p.parse_args()

    # ---- 1-3: the old checkpoint ------------------------------------------------------
    old, ck = T.load_checkpoint(ROOT / "runs" / CKPT)
    check("1 old checkpoint loads with anchor_omega=False",
          getattr(old, "anchor_omega", None) is False and "anchor_omega" not in ck["cfg"],
          f"model.anchor_omega = {getattr(old, 'anchor_omega', 'MISSING')!r}")

    data, meta = Dataset_Creator.load_dataset("famO_W40.npz")
    prep, W = T.prepare(data), meta["W"]
    _, va = T.group_split(prep["run_id"], 0.15, 0)
    runs = sorted(set(prep["run_id"][va].tolist()))[:150]
    try:
        n = rollout_rms(old, ck, prep, runs, W, anchor=False)
        check("2 old checkpoint, anchor off, scores as before",
              abs(n / F73_NORMAL - 1) < 1e-3, f"{n:.4e} vs {F73_NORMAL:.4e}")
        an = rollout_rms(old, ck, prep, runs, W, anchor=True)
        check("3 post-hoc anchor reproduces F73", abs(an / F73_ANCHOR - 1) < 1e-3,
              f"{an:.4e} vs {F73_ANCHOR:.4e}")
    except TypeError as e:
        check("2/3 predict_window takes anchor_omega=", False, str(e))

    # ---- 4-8: a fresh anchored model, with the real checkpoint's normalisation ----------
    try:
        m = fresh(True)
    except Exception as e:
        check("4-8 an anchored model can be built", False, repr(e))
        return summary()
    twin = fresh(False, like=m)
    rows = [runs[0] * W + 3, runs[1] * W + 17, runs[2] * W + 29]
    gstat = ck["gstat"]
    branch = T.build_branch({k: v[rows] for k, v in prep.items() if torch.is_tensor(v) and v.shape[:1] == prep["omega0"].shape[:1]},
                            ck["mu"], ck["sd"], gstat)
    B, t = len(rows), ck["t_local"]
    om0 = torch.tensor([-19.0, 0.3, 17.5])            # far from what the rows carry, on purpose
    kp, ki = prep["kp"][rows].view(-1, 1, 1), prep["ki"][rows].view(-1, 1, 1)

    # 4a inference path
    worst = 0.0
    for i, r in enumerate(rows):
        _, om = predict_window(m, ck, prep["theta0_abs"][r], om0[i], prep["Va"][r], prep["Vb"][r],
                               prep["Vc"][r], t, float(prep["kp"][r]), float(prep["ki"][r]))
        worst = max(worst, abs(float(om[0] - om0[i])))
    check("4a predict_window: omega(0) == omega0", worst < 1e-5, f"max |omega(0)-omega0| = {worst:.2e}")

    # 4b training path. The branch must carry the SAME omega0 for 5 to be a fair comparison.
    branch[:, 2] = (om0 - ck["mu"]) / ck["sd"]
    tq = t.view(1, -1, 1).expand(B, -1, 1).clone().requires_grad_(True)
    try:
        o = compute_theta_omega(m, tq, branch, torch.zeros(B, t.numel(), 1), omega_nominal=0.0,
                                Kp=kp, Ki=ki, gstat=gstat, omega0=om0)
    except TypeError as e:
        check("4b compute_theta_omega takes omega0=", False, str(e))
        return summary()
    d = float((o["omega"][:, 0, 0] - om0).abs().max())
    check("4b compute_theta_omega: omega(0) == omega0", d < 1e-5, f"max diff {d:.2e}")

    # 5 train == infer, on the same inputs
    worst_th = worst_om = 0.0
    for i, r in enumerate(rows):
        th, om = predict_window(m, ck, prep["theta0_abs"][r], om0[i], prep["Va"][r], prep["Vb"][r],
                                prep["Vc"][r], t, float(prep["kp"][r]), float(prep["ki"][r]))
        th_train = o["theta"][i, :, 0].detach() + prep["theta0_abs"][r] + T.OMEGA_BASE * t
        worst_th = max(worst_th, float((th - th_train).abs().max()))
        worst_om = max(worst_om, float((om - o["omega"][i, :, 0].detach()).abs().max()))
    check("5 training path == inference path", worst_th < 1e-4 and worst_om < 1e-4,
          f"theta {worst_th:.1e} rad, omega {worst_om:.1e} rad/s")

    # 6 domega/dt: with no limiter res_omega = domega/dt - Ki*Vq, and Vq=0 here, so it IS domega/dt
    tq2 = t.view(1, -1, 1).expand(B, -1, 1).clone().requires_grad_(True)
    o2 = compute_theta_omega(twin, tq2, branch, torch.zeros(B, t.numel(), 1), omega_nominal=0.0,
                             Kp=kp, Ki=ki, gstat=gstat, omega0=om0)
    d6 = float((o["res_omega"] - o2["res_omega"]).abs().max())
    check("6 domega/dt identical with and without the anchor", d6 < 1e-4, f"max diff {d6:.1e}")

    # 7 gradients reach every parameter through the anchored omega
    m.zero_grad()
    o["omega"].pow(2).mean().backward()
    bad = [nme for nme, prm in m.named_parameters() if prm.grad is None or not torch.isfinite(prm.grad).all()]
    check("7 gradients reach every parameter", not bad, f"missing/non-finite: {bad[:3]}" if bad else "")

    # 7b omega(0) is the constant omega0, so NO weight may move it. If N(0) was .detach()ed,
    # omega(0) = omega0 + N(0) - N(0).detach() still has a gradient -- and training would
    # push on a quantity that cannot change. Silent: it still trains, just wrongly.
    m.zero_grad()
    tq3 = t.view(1, -1, 1).expand(B, -1, 1).clone().requires_grad_(True)
    o3 = compute_theta_omega(m, tq3, branch, torch.zeros(B, t.numel(), 1), omega_nominal=0.0,
                             Kp=kp, Ki=ki, gstat=gstat, omega0=om0)
    o3["omega"][:, 0].sum().backward()
    g = max(float(prm.grad.abs().max()) for prm in m.parameters() if prm.grad is not None)
    check("7b omega(0) has zero gradient (N(0) not detached)", g < 1e-6, f"max |grad| {g:.1e}")

    # 8 save -> load
    with tempfile.TemporaryDirectory() as tmp:
        path = _Path(tmp) / "x.pth"
        T.save_checkpoint(path, m, meta, ck["mu"], ck["sd"], 1.0, 1.0, 1.0, t, {"train": [], "val": []}, 1,
                          0.3, gstat)
        m2, ck2 = T.load_checkpoint(path)
        r = rows[0]
        a1 = predict_window(m, ck, prep["theta0_abs"][r], om0[0], prep["Va"][r], prep["Vb"][r], prep["Vc"][r], t,
                            float(prep["kp"][r]), float(prep["ki"][r]))[1]
        a2 = predict_window(m2, ck2, prep["theta0_abs"][r], om0[0], prep["Va"][r], prep["Vb"][r], prep["Vc"][r], t,
                            float(prep["kp"][r]), float(prep["ki"][r]))[1]
        check("8 save -> load keeps anchor_omega and the outputs",
              getattr(m2, "anchor_omega", False) is True and ck2["cfg"].get("anchor_omega") is True
              and torch.allclose(a1, a2), f"loaded anchor_omega = {getattr(m2, 'anchor_omega', 'MISSING')!r}")

    # ---- 9: CLI ----------------------------------------------------------------------
    helptext = subprocess.run([_sys.executable, str(ROOT / "src" / "sweep.py"), "--help"],
                              capture_output=True, text=True).stdout
    check("9 --anchor_omega on sweep.py and in train_pll.main",
          "--anchor_omega" in helptext and "anchor_omega" in inspect.signature(T.main).parameters)

    # ---- 10: training smoke test ------------------------------------------------------
    if a.train:
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([_sys.executable, str(ROOT / "hpc" / "generate_family.py"), "--stem", "anch",
                            "--W", "40", "--n_runs", "20", "--lhs_seed", "99", "--freq_limit", "18.8496",
                            "--gains", "--outdir", tmp], check=True, capture_output=True)
            rec = T.main(dataset=str(_Path(tmp) / "anch_W40.npz"), epochs=2, w_phys=0.3, patience=40,
                         F=4, max_freq=503, n_layers=3, width=128, n_eval_runs=3, device="cpu",
                         results_dir=str(_Path(tmp) / "sw"), runs_dir=str(_Path(tmp) / "runs"),
                         anchor_omega=True)
            check("10 2-epoch training runs; tag and record carry the anchor",
                  rec.get("status") == "ok" and rec["tag"].endswith("_ao_g") and rec.get("anchor_omega") is True,
                  f"status={rec.get('status')} tag=...{rec['tag'][-14:]} anchor_omega={rec.get('anchor_omega')}")
    return summary()


def summary():
    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    _sys.exit(main())
