"""Where does the operator stop working? Evaluation only -- no training.

    python src/ood_test.py runs/famD_W40_n5000_W40_F4_mf503_wp0.3_s1sp0.pth

Every number in this project so far samples initial conditions from ONE box -- the LHS
ranges in config/initial_conditions.yml -- for training, validation, head_to_head and the
envelope figure alike. That measures interpolation thoroughly and extrapolation not at
all. Karampinis et al. make "generalization" a headline word, and we ourselves used the
extrapolation argument to restrict fig 12 to the paper NN's trained range, so the same
standard points back at us.

Two families of scenario, and the second is the one that matters:

  IC extrapolation   frequency offset, amplitude offset, omega_0 pushed 2-5x past their
                     trained ranges. Partly academic -- +/-1 Hz off nominal is a grid
                     that has bigger problems than its PLL -- but it locates the edge.
  FAULT extrapolation  sags DEEPER and LONGER than trained, jumps LARGER than trained.
                     This is NOT exotic: we train on 0.5-0.95 pu sags, and a bolted
                     three-phase fault is 0.0-0.3 pu. The realistic severe case sits
                     OUTSIDE the training box, which is exactly the case a transient
                     study cares about.

Design notes:
  * Scenarios are PAIRED. One set of uniforms is drawn per seed and then mapped into each
    scenario's ranges, so a scenario differs from the control by its range and by nothing
    else. Drawing fresh randoms per scenario would confound the comparison with sampling
    noise -- the same error that made the n_eval=20 results untrustworthy.
  * `solver @100us` is reported next to every scenario. It shares the surrogate's step,
    so it is the floor no method at this dt can beat. If the surrogate degrades and the
    solver does not, the network is extrapolating; if BOTH degrade, the scenario is
    simply harder to integrate and the network is not the problem.
  * The reference is the same 12.5 us trapezoidal solve used everywhere else (F41).
"""
import argparse

import numpy as np
import torch

from paths import ROOT
from speed_benchmark import solve_at, deeponet_at
from train_pll import load_checkpoint
from dataset_generator import Dataset_Creator

# trained ranges, from config/initial_conditions.yml -- the box everything so far sampled
TRAINED = dict(freq=0.2, amp=0.05, omega=20.0, eps=0.5 * torch.pi,
               sag_depth=(0.5, 0.95), sag_dur=(0.02, 0.10), jump_deg=60.0,
               fault_t0=(0.05, 0.40))

# name -> overrides. None of these touch anything not named, so one axis moves at a time.
SCENARIOS = {
    "in-distribution (control)": {},
    "freq offset x2  (+/-0.4 Hz)": dict(freq=0.4),
    "freq offset x5  (+/-1.0 Hz)": dict(freq=1.0),
    "amp offset x3   (+/-0.15 pu)": dict(amp=0.15),
    "omega_0 x2      (+/-40 rad/s)": dict(omega=40.0),
    "omega_0 x4      (+/-80 rad/s)": dict(omega=80.0),
    "sag DEEP        (0.1-0.5 pu)": dict(fault="sag", sag_depth=(0.10, 0.50)),
    "sag LONG        (0.1-0.3 s)": dict(fault="sag", sag_dur=(0.10, 0.30)),
    "jump BIG        (+/-120 deg)": dict(fault="jump", jump_deg=120.0),
    "sag trained     (0.5-0.95 pu)": dict(fault="sag"),      # fault control
    "jump trained    (+/-60 deg)": dict(fault="jump"),       # fault control
}


def build_case(u, dt_fine, horizon, over):
    """One scenario. `u` is the shared uniform draw (n_runs, 7) in [0,1) -- the SAME
    numbers for every scenario, mapped through different ranges."""
    torch.set_default_dtype(torch.float64)
    from PLL_Simulator import PLLSimulator

    p = {**TRAINED, **over}
    n_runs = u.shape[0]
    n_fine = int(round(horizon / dt_fine))
    sim = PLLSimulator(dt=dt_fine)
    sim.N, sim.n_runs = n_fine, n_runs
    sim.t = (torch.arange(n_fine) * dt_fine).reshape(1, n_fine)

    sym = lambda col, half: (u[:, col] * 2 - 1) * half          # U(-half, half)
    span = lambda col, lo_hi: lo_hi[0] + (lo_hi[1] - lo_hi[0]) * u[:, col]

    grid_ang = sym(0, torch.pi).unsqueeze(-1)
    freq_off = sym(1, p["freq"]).unsqueeze(-1)
    amp_off = sym(2, p["amp"]).unsqueeze(-1)
    theta0 = grid_ang.squeeze(-1) + sym(3, p["eps"])
    theta0 = (theta0 + torch.pi) % (2 * torch.pi) - torch.pi
    omega0 = sym(4, p["omega"])

    sag = jump = None
    if over.get("fault") == "sag":
        sag = (span(5, p["fault_t0"]).unsqueeze(-1),
               span(6, p["sag_dur"]).unsqueeze(-1),
               span(7, p["sag_depth"]).unsqueeze(-1))
    elif over.get("fault") == "jump":
        jump = (span(5, p["fault_t0"]).unsqueeze(-1),
                torch.deg2rad(sym(6, p["jump_deg"])).unsqueeze(-1))

    Va, Vb, Vc = sim._grid_phases(freq_off, amp_off, grid_ang, sag=sag, jump=jump)
    return dict(sim=sim, Va=Va, Vb=Vb, Vc=Vc, theta0=theta0, omega0=omega0,
                dt_fine=dt_fine, n_fine=n_fine, horizon=horizon)


def figure(groups, order, n_runs, out="15_ood_ladder.png"):
    """One ladder, ABSOLUTE error, every model family on the same axis.

    Absolute rather than ratio-to-control on purpose. A ratio hides the thing the dt
    comparison exists to show: famE's whole ladder sits 1.6x to the left of famD's (F48).
    Ratios also silently renormalise each family to its own control, which makes two
    families look identical when one is uniformly better.

    Per family: a solid bar (the surrogate) and a light bar (the trapezoidal solver at
    THAT family's step -- the floor no method at that dt can beat). Where a family has
    whole-turn disagreements, a hatched overlay shows what survives forgiving them."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from paths import GRAPHS

    names = list(groups)
    n_g = len(names)
    y = np.arange(len(order))
    h = 0.8 / (2 * n_g)                       # two bars (ours + solver) per family
    cols = ["tab:blue", "tab:purple", "tab:olive"]

    fig, ax = plt.subplots(figsize=(13.5, 1.05 + 0.62 * len(order) * n_g))
    for gi, gname in enumerate(names):
        res = groups[gname]
        base = 0.4 - (2 * gi + 1) * h
        ours = [res[k]["ours_abs"] for k in order]
        solv = [res[k]["solver_abs"] for k in order]
        ax.barh(y + base, ours, h, color=cols[gi % len(cols)], label=f"DeepONet  {gname}")
        ax.barh(y + base - h, solv, h, color="lightgray", edgecolor="dimgray", lw=0.5,
                label=f"trapezoid @ same step  {gname}")
        for i, k in enumerate(order):
            t, w = res[k]["turns"], res[k]["wrapped_abs"]
            if t:
                ax.barh(y[i] + base, w, h, color="white", edgecolor="tab:red",
                        hatch="///", lw=1.0, zorder=3)
            # annotation goes at the END of the longest bar in this row+family, so the
            # slip overlay and the value label can never land on top of each other.
            ax.annotate(f"  {res[k]['ours_abs']:.2e}"
                        + (f"   [{t}/{n_runs} slipped → {w:.2e}]" if t else ""),
                        (res[k]["ours_abs"], y[i] + base), va="center", ha="left",
                        fontsize=7, color="tab:red" if t else "black")

    ax.set_xscale("log")
    ax.set_xlim(right=max(r["ours_abs"] for g in groups.values() for r in g.values()) * 120)
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=8.5)
    ax.invert_yaxis()
    for i in range(len(order) - 1):           # separator between scenario blocks
        ax.axhline(i + 0.5, color="k", lw=0.4, alpha=0.25)
    ax.set_xlabel(r"absolute $\theta$ RMS vs a 12.5 $\mu$s trapezoidal reference [rad]  (log scale)")
    ax.grid(alpha=0.3, axis="x", which="both")
    hh, lab = ax.get_legend_handles_labels()
    if any(r["turns"] for g in groups.values() for r in g.values()):
        hh.append(Patch(facecolor="white", edgecolor="tab:red", hatch="///"))
        lab.append("same, forgiving whole cycle slips")
    ax.legend(hh, lab, fontsize=7.5, loc="lower right")
    ax.set_title("Out-of-distribution ladder — how far past the training box does it hold?\n"
                 f"{n_runs} runs x 0.5 s. Lower is better; the light bar is the "
                 f"discretisation floor at that family's own timestep.", fontsize=10)
    fig.tight_layout()
    GRAPHS.mkdir(exist_ok=True)
    fig.savefig(GRAPHS / out, dpi=160)
    print(f"\n-> {GRAPHS / out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", nargs="+", help="one or more checkpoints; the figure uses the first")
    p.add_argument("--n_runs", type=int, default=32)
    p.add_argument("--dt_fine", type=float, default=12.5e-6)
    p.add_argument("--horizon", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", default=None,
                   help="optional; by default W/S/dt are read from the checkpoint itself")
    p.add_argument("--out", default="15_ood_ladder.png", help="figure name under graphs/")
    a = p.parse_args()

    def meta_of(c):
        # save_checkpoint stores the dataset meta inside the .pth, so the multi-GB npz
        # does not have to be on this machine at all. That is what lets a famE (dt=50 us)
        # checkpoint be scored here without shipping its 2 GB dataset back from the cluster.
        p_ = ROOT / c if not c.startswith("/") else c
        return torch.load(p_, map_location="cpu", weights_only=False)["data_meta"]

    # Checkpoints from DIFFERENT families (different dt/W) may be mixed on one command
    # line: group them, and give each group its own solver floor and its own decimation.
    fams = {}
    for c in a.ckpt:
        m = Dataset_Creator.load_dataset(a.dataset)[1] if a.dataset else meta_of(c)
        key = f"{m['S']*m['W']} sensors, dt={m['dt']*1e6:.0f} us, W={m['W']}"
        fams.setdefault(key, {"meta": m, "ckpts": []})["ckpts"].append(c)

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(a.seed)
    u = torch.rand(a.n_runs, 8)                     # drawn ONCE, shared by every scenario

    print(f"OOD ladder  |  {a.n_runs} runs x {a.horizon} s, "
          f"reference = {a.dt_fine*1e6:.1f} us trapezoid")
    for k, v in fams.items():
        print(f"   family: {k}   ({len(v['ckpts'])} checkpoint(s))")
    print()
    hdr = f"  {'scenario':30s}"
    for k in fams:
        hdr += f" {'ours [' + k.split(',')[1].strip() + ']':>20s} {'floor':>10s}"
    print(hdr + f" {'turns':>7s}")

    groups = {k: {} for k in fams}
    order = []
    for name, over in SCENARIOS.items():
        # Re-seed so every scenario sees the SAME sensor-noise realisation. Without this
        # the noise depends on how many models were constructed earlier in the process --
        # load_checkpoint builds an MLP, which draws from the global RNG -- so adding a
        # second checkpoint to the command line silently changed the physics of every
        # row. Paired ICs (u) were never enough on their own.
        torch.manual_seed(a.seed + 1)
        case = build_case(u, a.dt_fine, a.horizon, over)
        ref_fine, _ = solve_at(case, a.dt_fine)
        order.append(name)
        row = f"  {name:30s}"

        for key, fam in fams.items():
            m = fam["meta"]
            W, S, dt_c = m["W"], m["S"], m["dt"]
            kc = int(round(dt_c / a.dt_fine))
            ref = ref_fine[:, ::kc][:, :W * S]
            th_sv, _ = solve_at(case, dt_c)
            e_sv = float((th_sv[:, :W * S] - ref).pow(2).mean().sqrt())

            e_nns, turns, e_wrapped = [], 0, 0.0
            for c in fam["ckpts"]:
                torch.set_default_dtype(torch.float32)
                model, ck = load_checkpoint(ROOT / c if not c.startswith("/") else c)
                th_nn, _ = deeponet_at(case, model, ck, W, S, dt_c)
                torch.set_default_dtype(torch.float64)
                err = th_nn.double() - ref
                e_nns.append(float(err.pow(2).mean().sqrt()))
                if c == fam["ckpts"][0]:
                    # wrapped error: what survives forgiving whole-turn disagreements.
                    # F43 shows the REFERENCE loop itself slips 5 cycles at omega_0 = 80
                    # and does not lock until 0.856 s, so past that point "which turn" is
                    # a discrete outcome of a slipping transient, not a tracking error.
                    ew = (err + np.pi) % (2 * np.pi) - np.pi
                    e_wrapped = float(ew.pow(2).mean().sqrt())
                    turns = int(((err[:, -1] / (2 * np.pi)).round() != 0).sum())
            groups[key][name] = dict(ours_abs=e_nns[0], solver_abs=e_sv,
                                     wrapped_abs=e_wrapped, turns=turns,
                                     all_seeds=e_nns)
            row += f" {e_nns[0]:20.3e} {e_sv:10.3e}"
        print(row + f" {turns:6d}/{a.n_runs}", flush=True)

    figure(groups, order, a.n_runs, a.out)

    print("\nturns off  = runs whose FINAL error is a whole multiple of 2*pi away, i.e. the")
    print("             surrogate counted cycle slips differently from the reference. That")
    print("             is a discrete failure worth 2*pi each, not a loss of precision.")
    print("ours/ctrl  = degradation of the surrogate vs its in-distribution control")
    print("solv/ctrl  = how much harder the scenario is to INTEGRATE at all -- the part")
    print("             that is physics, not extrapolation")
    print("ours/solv  = the surrogate's excess over the floor at its own step. This is")
    print("             the number that isolates extrapolation: if it stays near the")
    print("             control's value the operator is still doing its job, whatever")
    print("             happened to the absolute error.")


if __name__ == "__main__":
    main()
