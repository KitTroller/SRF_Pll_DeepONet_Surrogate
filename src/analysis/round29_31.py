"""exp29 (famQ, 2x data), exp30 (omega anchor), exp31 (famO70, jumps to +/-70): one common test.

    python src/analysis/round29_31.py            # both parts, ~5 min on a laptop CPU

PART 1 -- every model on the SAME 150 famO validation runs. Fair for all of them: famO's
val runs were never trained on by famO models (validation), by famQ models (a different
LHS draw), or by famO70 models (their paired twins sit in famO70's own val split).
Each model is scored as trained AND with the omega anchor applied after the fact, so
"trained under the anchor" and "anchored afterwards" meet on one axis.

PART 2 -- the phase-jump sweep, exp31's actual question. Fresh LIMITED truth (famO's
physics: limiter on, noise on, nominal Kp/Ki), one jump at t0 = 0.20 s, the SAME initial
conditions at every angle (seeded), signs alternating across runs. Peak and RMS error
against jump size, per model group.

Groups: base_cpu = exp28's 8 famO seeds; base_gpu = exp30's 4 unanchored GPU seeds (8-11);
anchor = exp30's 8 trained-anchored seeds; famQ = exp29; famO70 = exp31.
"""

# src/ on the path: these scripts live in src/analysis/ but import the pipeline
# modules (paths, PLL_Simulator, train_pll, sweep) that stay in src/. Running
# `python src/analysis/foo.py` puts src/analysis on sys.path, not src/.
# Same pattern as hpc/generate_family.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import argparse
import json
import statistics as st

import numpy as np
import torch

import PLL_Simulator as PS
from common_test import load_f32
from dataset_generator import Dataset_Creator
from paths import ROOT, graphs as _graphs
from pll_infer import predict_window, _gains
from train_pll import prepare, group_split

LIMIT = 18.8496
T = "_W40_n{n}_W40_F4_mf503_wp0.3_s{s}sp0_L3_w128{x}_g.pth"
GROUPS = {
    "base_cpu": [f"famO{T.format(n=5000, s=s, x='')}" for s in range(8)],
    "base_gpu": [f"famO{T.format(n=5000, s=s, x='')}" for s in range(8, 12)],
    "anchor":   [f"famO{T.format(n=5000, s=s, x='_ao')}" for s in range(8)],
    "famQ":     [f"famQ{T.format(n=10000, s=s, x='')}" for s in range(8)],
    "famO70":   [f"famO70{T.format(n=5000, s=s, x='')}" for s in range(8)],
}


def load(name):
    m, ck = load_f32(ROOT / "runs" / name)
    return m, ck


# ---------------------------------------------------------------- PART 1
def part1(n_runs, dataset="famO_W40.npz"):
    data, meta = Dataset_Creator.load_dataset(dataset)
    torch.set_default_dtype(torch.float32)
    prep, W = prepare(data), meta["W"]
    z = np.load(ROOT / "data" / dataset)
    kind, ang = z["fault_kind"][::W], np.degrees(np.abs(z["disturbance"][:, 4]))
    _, va = group_split(prep["run_id"], 0.15, 0)
    runs = sorted(set(prep["run_id"][va].tolist()))[:n_runs]
    big = [i for i, r in enumerate(runs) if kind[r] == 2 and ang[r] >= 40]

    out = {}
    for g, names in GROUPS.items():
        for name in names:
            model, ck = load(name)
            t = ck["t_local"]; t_ext = torch.cat([t, t[-1:] + (t[1] - t[0])])
            modes = [False] if getattr(model, "anchor_omega", False) else [False, True]
            for anchor in modes:
                th_rms, th_max, om_rms = [], [], []
                with torch.no_grad():
                    for r in runs:
                        row0 = r * W
                        th0, om0 = prep["theta0_abs"][row0], prep["omega0"][row0]
                        pth, pom = [], []
                        for k in range(W):
                            kp, ki = _gains(prep, row0 + k)
                            th, om = predict_window(model, ck, th0, om0, prep["Va"][row0 + k],
                                                    prep["Vb"][row0 + k], prep["Vc"][row0 + k],
                                                    t_ext, kp, ki, anchor_omega=anchor)
                            pth.append(th[:-1]); pom.append(om[:-1]); th0, om0 = th[-1], om[-1]
                        e = torch.cat(pth) - torch.cat([prep["theta_abs"][row0 + k] for k in range(W)])
                        eo = torch.cat(pom) - torch.cat([prep["target_omega"][row0 + k] for k in range(W)])
                        th_rms.append(float(e.pow(2).mean().sqrt())); th_max.append(float(e.abs().max()))
                        om_rms.append(float(eo.pow(2).mean().sqrt()))
                key = g + ("+posthoc" if anchor else "")
                out.setdefault(key, []).append({
                    "rms": float(np.mean(th_rms)), "max": float(np.max(th_max)),
                    "om_rms": float(np.mean(om_rms)),
                    "bigjump_peak": float(np.median([th_max[i] for i in big])),
                })
        print(f"  part 1: {g} done", flush=True)
    return out, len(big)


# ---------------------------------------------------------------- PART 2
def jump_truth(angles_deg, n_runs, seed=0):
    """famO physics with one jump per run at t0 = 0.20 s; same ICs for every angle."""
    PS.pll_constants.freq_limit = LIMIT
    PS.initial_conditions_config.white_noise_on_flag = True
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    dt, n = 100e-6, 5000
    sim = PS.PLLSimulator(dt=dt)
    sim.N, sim.n_runs = n, n_runs
    sim.t = (torch.arange(n) * dt).reshape(1, n)
    kp, ki = float(sim.physics.Kp), float(sim.physics.Ki)
    sim.physics.Kp = torch.full((n_runs,), kp, dtype=torch.float64)
    sim.physics.Ki = torch.full((n_runs,), ki, dtype=torch.float64)
    ga = (torch.rand(n_runs, 1) * 2 - 1) * torch.pi
    fo = (torch.rand(n_runs, 1) * 2 - 1) * 0.2
    ao = (torch.rand(n_runs, 1) * 2 - 1) * 0.05
    th0 = ga.squeeze(-1) + (torch.rand(n_runs) * 2 - 1) * 0.5 * torch.pi
    th0 = (th0 + torch.pi) % (2 * torch.pi) - torch.pi
    om0 = (torch.rand(n_runs) * 2 - 1) * 2.0
    state = torch.get_rng_state()                   # the noise draw is identical per angle
    sign = torch.where(torch.arange(n_runs) % 2 == 0, 1.0, -1.0).double().reshape(-1, 1)
    res = {}
    for a in angles_deg:
        torch.set_rng_state(state)
        jump = (torch.full((n_runs, 1), 0.20, dtype=torch.float64), sign * np.deg2rad(a))
        Va, Vb, Vc = sim._grid_phases(fo, ao, ga, jump=jump, sag=None)
        with torch.no_grad():
            th, om = sim.simulate_batch(Va, Vb, Vc, th0, om0, scheme="trapezoid")[:2]
        res[a] = (Va, Vb, Vc, th, om)
    return res, kp, ki


def roll(model, ck, V, th0, om0, kp, ki, anchor):
    """common_test.rollout, plus the post-hoc anchor switch it does not have."""
    W, S = ck["data_meta"]["W"], ck["data_meta"]["S"]
    t = ck["t_local"]; t_ext = torch.cat([t, t[-1:] + float(t[1] - t[0])])
    th = []
    for k in range(W):
        sl = slice(k * S, (k + 1) * S)
        p, o = predict_window(model, ck, th0, om0, V[0][sl].float(), V[1][sl].float(),
                              V[2][sl].float(), t_ext, kp, ki, anchor_omega=anchor)
        th.append(p[:-1]); th0, om0 = p[-1], o[-1]
    return torch.cat(th)


def part2(angles, n_runs):
    truth, kp, ki = jump_truth(angles, n_runs)
    torch.set_default_dtype(torch.float32)
    out = {}
    for g, names in GROUPS.items():
        for name in names:
            model, ck = load(name)
            modes = [False] if getattr(model, "anchor_omega", False) else [False, True]
            for anchor in modes:
                per = {}
                for a in angles:
                    Va, Vb, Vc, th, om = truth[a]
                    pk, rm = [], []
                    with torch.no_grad():
                        for r in range(n_runs):
                            pth = roll(model, ck, (Va[r], Vb[r], Vc[r]), th[r, 0].float(),
                                       om[r, 0].float(), kp, ki, anchor)
                            e = pth.double() - th[r]
                            pk.append(float(e.abs().max())); rm.append(float(e.pow(2).mean().sqrt()))
                    per[a] = {"peak": float(np.median(pk)), "rms": float(np.median(rm))}
                out.setdefault(g + ("+posthoc" if anchor else ""), []).append(per)
        print(f"  part 2: {g} done", flush=True)
    return out


def plot(path, out="29_round29_31.png"):
    """graphs/29 from the saved JSON -- no recompute."""
    import matplotlib.pyplot as plt
    d = json.load(open(path)); p1, p2, angles = d["part1"], d["part2"], d["angles"]
    GREY, BLUE, ORANGE, AQUA = "#555555", "#2a78d6", "#eb6834", "#1baf7a"
    cols = [("base_cpu", "baseline\n(deliverable)", GREY, "o"),
            ("base_gpu", "baseline\nGPU control", "#9a9a9a", "o"),
            ("base_cpu+posthoc", "baseline +\nanchor after", BLUE, "o"),
            ("anchor", "anchor\ntrained in", BLUE, "s"),
            ("famQ", "famQ\n2x data", ORANGE, "o"),
            ("famQ+posthoc", "famQ +\nanchor after", ORANGE, "D"),
            ("famO70", "famO70\njumps to 70", AQUA, "o")]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    rng = np.random.default_rng(0)
    b = st.median(r["rms"] for r in p1["base_cpu"])
    lo, hi = min(r["rms"] for r in p1["base_cpu"]), max(r["rms"] for r in p1["base_cpu"])
    ax[0].axhspan(lo, hi, color=GREY, alpha=.08, lw=0)
    for i, (k, lbl, c, m) in enumerate(cols):
        v = [r["rms"] for r in p1[k]]
        ax[0].scatter(i + rng.uniform(-.13, .13, len(v)), v, s=46, color=c, marker=m,
                      edgecolor="white", linewidth=1.1, zorder=3)
        med = st.median(v)
        ax[0].hlines(med, i - .27, i + .27, color="#222222", lw=2, zorder=4)
        if k != "base_cpu":
            ax[0].annotate(f"{b / med:.2f}x", (i, max(v)), xytext=(0, 6), textcoords="offset points",
                           ha="center", fontsize=9, color="#222222")
    ax[0].set_xticks(range(len(cols)))
    ax[0].set_xticklabels([f"{l}\nn={len(p1[k])}" for k, l, _, _ in cols], fontsize=8.3)
    ax[0].set_ylabel("deployed theta RMS on famO's 150 val runs [rad]", fontsize=9)
    ax[0].set_title("Common test: every model on the same held-out runs\n"
                    "number = baseline median / group median (>1 is better)", fontsize=10)
    for j, (metric, lab) in enumerate([("peak", "median per-run PEAK |theta error| [rad]"),
                                       ("rms", "median per-run theta RMS [rad]")], start=1):
        a = ax[j]
        a.axvspan(0, 60, color=GREY, alpha=.07, lw=0)
        a.axvline(60, color=GREY, lw=.8, ls=":"); a.axvline(70, color=AQUA, lw=.8, ls=":")
        for k, lbl, c, ls in [("base_cpu", "baseline", GREY, "-"),
                              ("base_cpu+posthoc", "baseline + anchor after", BLUE, "-"),
                              ("famQ+posthoc", "famQ + anchor after", ORANGE, "-"),
                              ("famO70", "famO70 (trained to 70 deg)", AQUA, "--")]:
            curves = np.array([[s[str(x)][metric] for x in angles] for s in p2[k]])
            a.plot(angles, np.median(curves, 0), color=c, ls=ls, lw=2, marker="o", ms=4, label=lbl)
            a.fill_between(angles, curves.min(0), curves.max(0), color=c, alpha=.12, lw=0)
        a.set_xlabel("phase-jump angle [deg]   grey: famO trained range")
        a.set_ylabel(lab, fontsize=9); a.grid(alpha=.25)
        a.set_title(("Peak error vs jump size: no edge effect at 60 deg,\nand widening the range buys nothing"
                     if metric == "peak" else "RMS vs jump size"), fontsize=10)
        a.legend(fontsize=7.8, loc="upper left")
    for a in ax: a.spines[["top", "right"]].set_visible(False)
    ax[0].grid(alpha=.25, axis="y")
    fig.suptitle("exp29-31 on one axis. The data and the anchor each buy 1.2-1.3x and they stack "
                 "(famQ + anchor after = %.2fx); training WITH the anchor does not beat adding it "
                 "afterwards; wider jumps change nothing.\nPart 2: limited truth, nominal Kp/Ki, "
                 "%d runs per angle, same initial conditions at every angle; bands = seed range."
                 % (b / st.median(r["rms"] for r in p1["famQ+posthoc"]), d["jump_runs"]), fontsize=10.5)
    fig.tight_layout()
    o = _graphs(_Path(out)); o.parent.mkdir(exist_ok=True); fig.savefig(o, dpi=160)
    print(f"-> {o}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plot_only", action="store_true", help="redraw graphs/29 from the saved JSON")
    p.add_argument("--split", default="famO_W40.npz",
                   help="whose validation runs are the common test for PART 1. famQ_W40.npz is "
                        "exp29's pre-registered SECOND split; with it, part 2 and the plot are skipped")
    p.add_argument("--n_runs", type=int, default=150)
    p.add_argument("--jump_runs", type=int, default=32)
    p.add_argument("--angles", type=float, nargs="+", default=[20, 40, 50, 55, 60, 65, 70, 75])
    p.add_argument("--out", default=str(ROOT / "Hyperparameter_sweep" / "round29_31.json"))
    a = p.parse_args()
    if a.plot_only:
        return plot(a.out)

    p1, n_big = part1(a.n_runs, a.split)
    if a.split != "famO_W40.npz":
        a.out = a.out.replace(".json", f"_split_{_Path(a.split).stem}.json")
    print(f"\nPART 1 -- all models on {a.split}'s {a.n_runs} val runs ({n_big} with a jump >= 40 deg). Medians over seeds [min-max]:")
    base = st.median(r["rms"] for r in p1["base_cpu"])
    print(f"{'group':20s} {'n':>2s}  {'theta RMS':>26s}  {'x base':>6s}  {'worst case':>22s}  {'omega RMS':>9s}  {'big-jump peak':>13s}")
    for k, v in p1.items():
        f = lambda m: (st.median(r[m] for r in v), min(r[m] for r in v), max(r[m] for r in v))
        rm, mx, om, bj = f("rms"), f("max"), f("om_rms"), f("bigjump_peak")
        print(f"{k:20s} {len(v):2d}  {rm[0]:.3e} [{rm[1]:.2e}-{rm[2]:.2e}]  {base / rm[0]:6.2f}  "
              f"{mx[0]:.4f} [{mx[1]:.4f}-{mx[2]:.4f}]  {om[0]:.2e}  {bj[0]:.2e}")

    if a.split != "famO_W40.npz":
        json.dump({"part1": p1, "split": a.split, "n_runs": a.n_runs}, open(a.out, "w"), indent=1)
        return print(f"\n-> {a.out}")
    p2 = part2(a.angles, a.jump_runs)
    print(f"\nPART 2 -- phase-jump sweep, limited truth, {a.jump_runs} runs per angle, median per-run PEAK |theta error| [rad], median over seeds:")
    print(f"{'angle':>6s} " + " ".join(f"{g:>10s}" for g in p2))
    for ang in a.angles:
        print(f"{ang:6.0f} " + " ".join(f"{st.median(s[ang]['peak'] for s in v):10.2e}" for v in p2.values()))
    print("\n... and median per-run RMS:")
    for ang in a.angles:
        print(f"{ang:6.0f} " + " ".join(f"{st.median(s[ang]['rms'] for s in v):10.2e}" for v in p2.values()))

    json.dump({"part1": p1, "part2": {g: [{str(k): v for k, v in s.items()} for s in seeds]
                                      for g, seeds in p2.items()},
               "angles": a.angles, "n_runs": a.n_runs, "jump_runs": a.jump_runs},
              open(a.out, "w"), indent=1)
    print(f"\n-> {a.out}")
    plot(a.out)


if __name__ == "__main__":
    main()
