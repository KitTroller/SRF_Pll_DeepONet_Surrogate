"""One config per process. Safe to run many terminals at once.

    python sweep.py --w_phys 0.3
    python sweep.py --F 4 --max_freq 1885 --w_phys 0.3 --seed 1 --results_dir sweeps_ff
    python sweep.py --collect --results_dir sweeps_wphys --plot wphys
    python sweep.py --collect --results_dir sweeps_ff    --plot ff
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse, json, collections
from pathlib import Path
import numpy as np

from paths import sweeps as _sweeps, graphs as _graphs

FLOOR_R1, FLOOR_R2 = 9.459e-3, 1.090e-2      # (Kp*noise/s1)^2 and (Ki*noise/s2)^2


def load(results_dir, keep_diverged=False):
    recs = [json.loads(p.read_text()) for p in _sweeps(results_dir).glob("*.json")]
    bad = [r for r in recs if r.get("status", "ok") != "ok"]
    if bad and not keep_diverged:
        print(f"!! {len(bad)} diverged run(s) EXCLUDED: {', '.join(r['tag'] for r in bad)}")
        recs = [r for r in recs if r.get("status", "ok") == "ok"]
    return sorted(recs, key=lambda r: (r["W"], r["F"], r.get("max_freq", 0),
                                       r["w_phys"], r["seed"]))


def table(recs):
    if not recs:
        print("no results"); return
    hdr = (f"{'tag':56s} {'roll_rms':>10s} {'roll_max':>10s} {'per_win':>10s} {'comp':>6s} "
           f"{'val_th':>10s} {'val_om':>10s} {'r1':>9s} {'r2':>9s} {'ep':>4s} {'ran':>4s} {'s':>6s}")
    print(hdr); print("-" * len(hdr))
    for r in recs:
        print(f"{r['tag']:56s} {r['rollout_full_rms']:10.3e} {r['rollout_full_max']:10.3e} "
              f"{r['per_window_rms']:10.3e} {r['compounding']:6.2f} {r['val_th']:10.3e} "
              f"{r['val_om']:10.3e} {r['r1']:9.3e} {r['r2']:9.3e} {r['best_epoch']:4d} "
              f"{r['epochs_run']:4d} {r['seconds']:6.0f}")
    b = min(recs, key=lambda r: r["rollout_full_rms"])
    print(f"\nbest by rollout_full_rms: {b['tag']}  {b['rollout_full_rms']:.4e} rad")


def _band(group, key):
    v = [r[key] for r in group]
    return float(np.median(v)), min(v), max(v)


def plot_wphys(recs, out="graphs/07_wphys_sweep.png", min_epochs=0):
    import matplotlib.pyplot as plt
    recs = [r for r in recs if r["epochs_run"] >= min_epochs]
    if not recs:
        print("nothing to plot"); return
    W0, F0 = recs[0]["W"], recs[0]["F"]
    recs = [r for r in recs if r["W"] == W0 and r["F"] == F0]
    by_w = collections.defaultdict(list)
    for r in recs:
        by_w[r["w_phys"]].append(r)
    xs   = sorted(k for k in by_w if k > 0)
    zero = by_w.get(0.0, [])
    ep   = sorted({r["epochs_run"] for r in recs})

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))

    # ---------------- left: the metric that matters ----------------
    m, lo, hi = (np.array(v) for v in zip(*[_band(by_w[k], "rollout_full_rms") for k in xs]))
    ax[0].errorbar(xs, m, yerr=[m - lo, hi - m], fmt="o-", capsize=4, lw=1.6,
                   color="tab:blue", label="median (bar = seed range)")
    if zero:
        z = [r["rollout_full_rms"] for r in zero]
        ax[0].axhline(np.median(z), color="k", ls="--", lw=1.3, label="$w_{phys}=0$")
        ax[0].axhspan(min(z), max(z), color="k", alpha=0.08)
    ax[0].set_ylabel("deployed $\\theta$ RMS @ 0.5 s  [rad]")
    ax[0].set_title("deployed error (40 handovers, no ground truth)")

    # ---------------- middle: residuals, explicitly ----------------
    for key, col, name, floor in [("r1", "tab:blue",   "eq 1:  $\\dot\\theta-\\omega-K_pV_q$", FLOOR_R1),
                                  ("r2", "tab:orange", "eq 2:  $\\dot\\omega-K_iV_q$",         FLOOR_R2)]:
        m, lo, hi = (np.array(v) for v in zip(*[_band(by_w[k], key) for k in xs]))
        ax[1].errorbar(xs, m, yerr=[m - lo, hi - m], fmt="o-", capsize=4, lw=1.6,
                       color=col, label=name)
        if zero:
            ax[1].axhline(np.median([r[key] for r in zero]), color=col, ls="--", lw=1.2,
                          label=f"  ...at $w_{{phys}}=0$")
        ax[1].axhline(floor, color=col, ls="-.", lw=1.4,
                      label=f"  ...sensor-noise floor")
    ax[1].set_ylabel("residual / (term RMS)$^2$   [dimensionless]")
    ax[1].set_title("residuals sit on a floor predicted in advance")

    # ---------------- right: teacher-forced theta ----------------
    m, lo, hi = (np.array(v) for v in zip(*[_band(by_w[k], "val_th") for k in xs]))
    ax[2].errorbar(xs, m, yerr=[m - lo, hi - m], fmt="o-", capsize=4, lw=1.6,
                   color="tab:blue", label="median (bar = seed range)")
    if zero:
        z = [r["val_th"] for r in zero]
        ax[2].axhline(np.median(z), color="k", ls="--", lw=1.3, label="$w_{phys}=0$")
        ax[2].axhspan(min(z), max(z), color="k", alpha=0.08)
    ax[2].set_ylabel("val MSE $\\theta$  [rad$^2$]")
    ax[2].set_title("teacher-forced $\\theta$ (one window, true IC)")

    for a in ax:
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlabel("$w_{phys}$"); a.grid(alpha=0.3, which="both")
        a.legend(fontsize=7, framealpha=0.9)
    fig.suptitle(f"Physics-weight sweep  (W={W0}, S={recs[0]['S']}, F={F0}, "
                 f"{len(recs)} runs, {min(ep)}-{max(ep)} epochs)")
    fig.tight_layout()
    out = _graphs(out); out.parent.mkdir(exist_ok=True); fig.savefig(out, dpi=160)
    print(f"-> {out}  ({len(recs)} runs, {len(zero)} at w_phys=0)")


def plot_ff(recs, out="graphs/08_fourier_sweep.png", min_epochs=0):
    """x = max Fourier frequency, one series per F. F=0 is the control band.
    Top axis = cycles the top feature completes inside ONE window -- the number
    that says whether the feature can do anything at all."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator
    recs = [r for r in recs if r["epochs_run"] >= min_epochs]
    if not recs:
        print("nothing to plot"); return
    win = recs[0]["window_s"]
    ctrl = [r for r in recs if r["F"] == 0]
    by = collections.defaultdict(list)
    for r in recs:
        if r["F"] > 0:
            by[(r["F"], r["max_freq"])].append(r)
    Fs = sorted({k[0] for k in by})

    panels = [("rollout_full_rms", "deployed $\\theta$ RMS @ 0.5 s [rad]", "deployed error"),
              ("per_window_rms",   "one-window $\\theta$ RMS [rad]",       "operator quality"),
              ("val_th",           "val MSE $\\theta$ [rad$^2$]",          "teacher-forced $\\theta$")]
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))
    for a, (key, lab, title) in zip(ax, panels):
        for i, F in enumerate(Fs):
            xs = sorted(mf for (f, mf) in by if f == F)
            m, lo, hi = (np.array(v) for v in zip(*[_band(by[(F, x)], key) for x in xs]))
            a.errorbar(xs, m, yerr=[m - lo, hi - m], fmt="o-", capsize=4, lw=1.5,
                       label=f"F = {F}")
        if ctrl:
            c = [r[key] for r in ctrl]
            a.axhline(np.median(c), color="k", ls="--", lw=1.3, label="F = 0 (control)")
            a.axhspan(min(c), max(c), color="k", alpha=0.08)
        a.set_xscale("log"); a.set_yscale("log")
        all_mf = sorted({mf for (_, mf) in by})
        a.set_xticks(all_mf)
        a.set_xticklabels([f"{x:g}\n{x*win/(2*np.pi):.2f} cyc" for x in all_mf], fontsize=7)
        a.xaxis.set_minor_locator(NullLocator())
        a.set_xlabel("max Fourier frequency [rad/s]  /  cycles per window")
        a.set_ylabel(lab)
        a.set_title(title); a.grid(alpha=0.3, which="major"); a.legend(fontsize=7)
    fig.suptitle(f"Fourier-feature sweep  (window = {win*1e3:.1f} ms, "
                 f"W={recs[0]['W']}, $w_{{phys}}$={recs[0]['w_phys']:g}, {len(recs)} runs)")
    fig.tight_layout()
    out = _graphs(out); out.parent.mkdir(exist_ok=True); fig.savefig(out, dpi=160)
    print(f"-> {out}  ({len(recs)} runs, {len(ctrl)} controls)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--collect", action="store_true")
    p.add_argument("--plot", choices=["wphys", "ff"], default="wphys")
    p.add_argument("--results_dir", default="sweeps")
    p.add_argument("--out", default=None)
    p.add_argument("--min_epochs", type=int, default=0)
    p.add_argument("--dataset", default="pll_dataset.npz")
    p.add_argument("--w_phys", type=float, default=0.0)
    p.add_argument("--F", type=int, default=None)
    p.add_argument("--max_freq", type=float, default=None)
    p.add_argument("--hidden_dim", type=int, default=None)
    p.add_argument("--n_layers", type=int, default=None,
                   help="interior DEPTH of both nets; default 2 reproduces the YAML")
    p.add_argument("--width", type=int, default=None,
                   help="interior WIDTH of both nets; default 64. Distinct from "
                        "--hidden_dim, which sets the LATENT contraction width")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--device", default=None)
    p.add_argument("--n_eval_runs", type=int, default=20)
    p.add_argument("--arch", choices=["deeponet", "pinn"], default="deeponet",
                   help="pinn = Single_PINN, the plain-MLP control. Tag gets _pinn.")
    p.add_argument("--residual", choices=["eq4", "eq6"], default="eq4",
                   help="eq4 = stored Vq (gauge invariant, pure derivative supervision); "
                        "eq6 = Vq recomputed from the predicted angle. Tag gets _eq6.")
    a = p.parse_args()

    if a.collect:
        recs = load(a.results_dir)
        table(recs)
        n = {"wphys": "07", "ff": "08"}[a.plot]
        out = a.out or f"graphs/{n}_{Path(a.results_dir).name}.png"
        (plot_wphys if a.plot == "wphys" else plot_ff)(recs, out=out, min_epochs=a.min_epochs)
    else:
        import train_pll as T
        T.main(dataset=a.dataset, epochs=a.epochs, lr=a.lr, w_phys=a.w_phys,
               batch_size=a.batch_size, patience=a.patience, seed=a.seed,
               split_seed=a.split_seed, F=a.F, max_freq=a.max_freq, hidden_dim=a.hidden_dim,
               device=a.device or T.DEVICE, n_eval_runs=a.n_eval_runs,
               results_dir=a.results_dir, arch=a.arch, residual=a.residual,
               n_layers=a.n_layers, width=a.width)