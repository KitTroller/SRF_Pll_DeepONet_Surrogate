"""
pll_plots.py -- every figure for the report. Run: python pll_plots.py
Writes PNGs into graphs/.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from dataset_generator import Dataset_Creator
from train_pll import prepare, group_split, load_checkpoint, OMEGA_BASE
from pll_infer import predict_window

GRAPHS = Path(__file__).resolve().parent / "graphs"
GRAPHS.mkdir(exist_ok=True)
DPI = 1024

wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi


def load_all(dataset="pll_dataset.npz", ckpt="pll_deeponet.pth"):
    data, meta = Dataset_Creator.load_dataset(dataset)
    prep = prepare(data)
    model, ck = load_checkpoint(ckpt)
    tr, va = group_split(prep["run_id"], 0.15, 0)
    return data, meta, prep, model, ck, tr, va


def stitch(arr, run_idx, W):
    """(n_samples, S) windowed  ->  (N,) one whole run in time order."""
    return torch.cat([arr[run_idx * W + k] for k in range(W)])


# ---------------------------------------------------------------- figure 1
def fig_initial_conditions(data, meta):
    """a-i: does the LHS actually cover the 5-D box we asked for?
    Left  = strip plot (every run as a point on its own axis)
    Right = histogram (LHS should be flat; this is the property Cartesian
            grids give you for free and LHS has to earn)"""
    lhs = data["lhs_samples"].numpy()
    cols, rng = meta["columns"], meta["ranges"]

    # the physically meaningful coordinate is the initial phase ERROR, not
    # theta_pll on its own -- it is what the `dependent` flag actually controls
    eps0 = wrap(lhs[:, 0] - lhs[:, 3])
    panels = [(cols[i], lhs[:, i], rng[cols[i]]) for i in range(5)]
    panels.append(("initial phase error  eps0 = grid - pll", eps0, [-np.pi, np.pi]))

    fig, ax = plt.subplots(6, 2, figsize=(11, 12),
                           gridspec_kw={"width_ratios": [3, 1]})
    for r, (name, v, (lo, hi)) in enumerate(panels):
        jitter = (np.random.rand(len(v)) - 0.5) * 0.6
        ax[r][0].scatter(v, jitter, s=3, alpha=0.35, lw=0)
        ax[r][0].axvspan(lo, hi, color="tab:green", alpha=0.07)
        ax[r][0].axvline(lo, color="tab:green", lw=1, ls="--")
        ax[r][0].axvline(hi, color="tab:green", lw=1, ls="--")
        ax[r][0].set_ylabel(name, fontsize=8)
        ax[r][0].set_yticks([]); ax[r][0].grid(alpha=0.3, axis="x")
        ax[r][0].set_title(f"min {v.min():+.3f}   max {v.max():+.3f}",
                           fontsize=7, loc="right")
        ax[r][1].hist(v, bins=40, color="tab:blue", alpha=0.7)
        ax[r][1].set_yticks([]); ax[r][1].grid(alpha=0.3)
    ax[0][0].set_title("configured range shaded; each dot is one run", fontsize=9)
    ax[5][0].set_xlabel("value"); ax[5][1].set_xlabel("count")
    fig.suptitle(f"Initial-condition sampling, {len(lhs)} runs, 5-D Latin Hypercube")
    fig.tight_layout()
    fig.savefig(GRAPHS / "01_initial_conditions.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_lock_check(data, meta, n_show=10):
    """a-ii + a-iii: does the simulated PLL actually lock?
    theta_err uses atan2(Vbeta, Valpha) as the reference -- an INDEPENDENT
    estimate of the grid angle that never touches the PLL loop."""
    W, dt, N = meta["W"], meta["dt"], meta["N"]
    t = np.arange(N) * dt
    lhs = data["lhs_samples"].numpy()
    runs = np.linspace(0, meta["n_runs"] - 1, n_show).astype(int)

    fig, ax = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for r in runs:
        th   = stitch(data["theta_pll"], r, W).numpy()
        om   = stitch(data["omega_pll"], r, W).numpy()
        vq   = stitch(data["Vq"],        r, W).numpy()
        va   = stitch(data["Valpha"],    r, W).numpy()
        vb   = stitch(data["Vbeta"],     r, W).numpy()

        th_ref = np.unwrap(np.arctan2(vb, va))       # Clarke -> grid angle, no PLL
        ax[0][0].plot(t, wrap(th - th_ref), lw=0.5)
        ax[0][1].plot(t, om, lw=0.5)
        # each run must settle at exactly 2*pi*frequency_offset
        ax[0][1].axhline(2 * np.pi * lhs[r, 1], color="k", lw=0.4, ls=":")
        ax[1][0].plot(t, vq, lw=0.3)
        ax[1][1].plot(t, np.sqrt(va**2 + vb**2), lw=0.5)

    ax[0][0].set_ylabel("theta_pll - atan2(Vbeta,Valpha)  [rad]")
    ax[0][0].set_title("phase error vs an independent angle reference")
    ax[0][0].axhline(0, color="k", lw=0.5)
    ax[0][1].set_ylabel("omega_pll  [rad/s]")
    ax[0][1].set_title("integrator state; dotted = 2*pi*frequency_offset target")
    ax[1][0].set_ylabel("Vq  [pu]"); ax[1][0].axhline(0, color="k", lw=0.5)
    ax[1][0].set_title("Vq -> 0 is the lock condition"); ax[1][0].set_xlabel("time [s]")
    ax[1][1].set_ylabel("|V| = sqrt(Valpha^2 + Vbeta^2)  [pu]")
    ax[1][1].set_title("envelope: amplitude offset + harmonics + noise")
    ax[1][1].set_xlabel("time [s]")
    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.suptitle(f"Simulated PLL, {n_show} runs")
    fig.tight_layout()
    fig.savefig(GRAPHS / "02_lock_check.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_prediction_vs_truth(prep, meta, model, ck, val_runs, n_show=10):
    """b: model vs simulator. theta is plotted in DEVIATION form -- the
    absolute angle ramps to ~157 rad and hides everything."""
    W, dt, S = meta["W"], meta["dt"], meta["S"]
    t_local = ck["t_local"]
    dt_ = float(t_local[1] - t_local[0])
    t_ext = torch.cat([t_local, t_local[-1:] + dt_])
    t_full = np.arange(W * S) * dt
    bounds = [k * S * dt for k in range(1, W)]
    runs = val_runs[:n_show]

    fig, ax = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    show = runs[0]
    for r in runs:
        row0 = r * W
        th0, om0 = prep["theta0_abs"][row0], prep["omega0"][row0]
        th_p, om_p = [], []
        for k in range(W):                                  # recurrent rollout
            th, om = predict_window(model, ck, th0, om0,
                                    prep["Va"][row0 + k], prep["Vb"][row0 + k],
                                    prep["Vc"][row0 + k], t_ext)
            th_p.append(th[:-1]); om_p.append(om[:-1])
            th0, om0 = th[-1], om[-1]                       # the feedback
        th_p = torch.cat(th_p).numpy(); om_p = torch.cat(om_p).numpy()

        th_t = stitch(prep["theta_abs"],   r, W).numpy()
        om_t = stitch(prep["target_omega"], r, W).numpy()
        dev  = prep["theta0_abs"][row0].item() + OMEGA_BASE * t_full

        if r == show:
            ax[0][0].plot(t_full, th_t - dev, lw=1.0, label="simulator")
            ax[0][0].plot(t_full, th_p - dev, lw=1.0, ls="--", label="DeepONet")
            ax[1][0].plot(t_full, om_t, lw=1.0, label="simulator")
            ax[1][0].plot(t_full, om_p, lw=1.0, ls="--", label="DeepONet")
        ax[0][1].plot(t_full, th_p - th_t, lw=0.6)
        ax[1][1].plot(t_full, om_p - om_t, lw=0.6)

    for a in ax.ravel():
        for b in bounds:
            a.axvline(b, color="grey", lw=0.5, ls=":")      # window handover
        a.grid(alpha=0.3)
    ax[0][0].set_ylabel("theta deviation [rad]"); ax[0][0].legend(fontsize=8)
    ax[0][0].set_title(f"run {show}: predicted vs simulated")
    ax[0][1].set_ylabel("theta error [rad]"); ax[0][1].axhline(0, color="k", lw=0.5)
    ax[0][1].set_title(f"{len(runs)} validation runs; dotted = window handover")
    ax[1][0].set_ylabel("omega [rad/s]"); ax[1][0].legend(fontsize=8)
    ax[1][0].set_xlabel("time [s]")
    ax[1][1].set_ylabel("omega error [rad/s]"); ax[1][1].axhline(0, color="k", lw=0.5)
    ax[1][1].set_xlabel("time [s]")
    fig.suptitle("Recurrent rollout: only the STATE is fed back, Va/Vb/Vc stay true")
    fig.tight_layout()
    fig.savefig(GRAPHS / "03_prediction_vs_truth.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def fig_window_sweep(prep, meta, model, ck, val_runs, n_runs_avg=20):
    """c: the window sweep, WITH the control you are missing.
    'fed back'  = each window starts from the PREDICTED previous state
    'true IC'   = each window starts from the TRUE state
    The gap between them is compounding; the 'true IC' line is the model's
    own one-window error and is what n=1 measures."""
    W, S, dt = meta["W"], meta["S"], meta["dt"]
    t_local = ck["t_local"]
    dt_ = float(t_local[1] - t_local[0])
    t_ext = torch.cat([t_local, t_local[-1:] + dt_])
    ns = range(1, W + 1)
    fed, tru = [], []

    for n in ns:
        e_f, e_t = [], []
        for r in val_runs[:n_runs_avg]:
            row0 = r * W
            th0, om0 = prep["theta0_abs"][row0], prep["omega0"][row0]
            pf, pt = [], []
            for k in range(n):
                th, om = predict_window(model, ck, th0, om0, prep["Va"][row0+k],
                                        prep["Vb"][row0+k], prep["Vc"][row0+k], t_ext)
                pf.append(th[:-1]); th0, om0 = th[-1], om[-1]
                th2, _ = predict_window(model, ck,
                                        prep["theta0_abs"][row0+k], prep["omega0"][row0+k],
                                        prep["Va"][row0+k], prep["Vb"][row0+k],
                                        prep["Vc"][row0+k], t_ext)
                pt.append(th2[:-1])
            truth_th = torch.cat([prep["theta_abs"][row0+k] for k in range(n)])
            e_f.append(((torch.cat(pf) - truth_th) ** 2).mean().sqrt().item())
            e_t.append(((torch.cat(pt) - truth_th) ** 2).mean().sqrt().item())
        fed.append(np.mean(e_f)); tru.append(np.mean(e_t))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ns, fed, "o-", label="state fed back (recurrent)")
    ax[0].plot(ns, tru, "s--", label="true IC each window (no feedback)")
    ax[0].set_yscale("log"); ax[0].set_xlabel("windows rolled out")
    ax[0].set_ylabel("theta RMS error [rad]"); ax[0].legend(fontsize=8)
    ax[0].set_title(f"mean over {n_runs_avg} validation runs")
    ax[1].plot(ns, np.array(fed) / np.array(tru), "o-", color="tab:red")
    ax[1].axhline(1, color="k", lw=0.5)
    ax[1].set_xlabel("windows rolled out"); ax[1].set_ylabel("compounding factor")
    ax[1].set_title("how much of the error is feedback, not the model")
    for a in ax: a.grid(alpha=0.3)
    fig.suptitle(f"Window sweep (W = {W}; beyond {W} you walk into the next RUN)")
    fig.tight_layout()
    fig.savefig(GRAPHS / "04_window_sweep.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- figure 5
def fig_error_by_window(data, prep, meta, model, ck, tr, va):
    """my addition: a single scalar val loss hides that ~2/3 of the error is
    in window 0. R^2 = 1 - MSE/var; R^2 <= 0 means 'no better than the mean'."""
    from torch.utils.data import DataLoader
    from train_pll import PLLDataset, assemble_batch
    from pll_residual import compute_theta_omega
    W, seg = meta["W"], data["segment_id"]

    def stats(mask):
        dl = DataLoader(PLLDataset(prep, mask), batch_size=256)
        TH, OM, TT, TO = [], [], [], []
        for b in dl:
            br, tq, Vq, tth, tom = assemble_batch(b, prep["t_local"], ck["mu"], ck["sd"])
            o = compute_theta_omega(model, tq, br, Vq, omega_nominal=0.0)
            TH.append(o["theta"].detach()); OM.append(o["omega"].detach())
            TT.append(tth); TO.append(tom)
        th, om, tt, to = map(torch.cat, (TH, OM, TT, TO))
        return (1 - (th-tt).pow(2).mean()/tt.var()).item(), \
               (1 - (om-to).pow(2).mean()/to.var()).item(), \
               (th-tt).pow(2).mean().item()

    r2t, r2o, mse = zip(*[stats(va & (seg == w)) for w in range(W)])
    x = np.arange(W)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(x - 0.2, r2t, 0.4, label="R^2 theta")
    ax[0].bar(x + 0.2, r2o, 0.4, label="R^2 omega")
    ax[0].axhline(0, color="k", lw=0.5); ax[0].set_ylim(-0.1, 1.05)
    ax[0].set_xlabel("window index"); ax[0].legend(fontsize=8)
    ax[0].set_title("validation R^2 per window")
    share = np.array(mse) / np.sum(mse) * 100
    ax[1].bar(x, share, color="tab:orange")
    ax[1].set_xlabel("window index"); ax[1].set_ylabel("% of total theta error")
    ax[1].set_title("where the error budget is spent")
    for a in ax: a.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(GRAPHS / "05_error_by_window.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- figure 6
def fig_residual_budget(data, meta):
    """my addition: the picture behind 'why is ph 1e4'.
    residual = d2(theta)/dt2 - Ki*Vq, but the true relation is
    d2(theta)/dt2 = Ki*Vq + Kp*dVq/dt. Autograd cannot see d(Vq)/dt because Vq
    is stored data, so Kp*dVq/dt is silently dropped -- and it is the BIGGER
    of the two terms."""
    dt, Ki, Kp, W = meta["dt"], meta["Ki"], meta["Kp"], meta["W"]
    S = meta["S"]
    k    = max(3, min(51, (S // 10) | 1))   # odd kernel, never longer than S/10
    trim = max(1, k)                        # drop the convolution edges only
    seg = data["segment_id"]
    ker = np.ones(51) / 51
    kept, dropped, noise = [], [], []
    for w in range(W):
        Vq = data["Vq"][seg == w].numpy().astype(np.float64)
        Vqs = np.apply_along_axis(lambda r: np.convolve(r, ker, "same"), 1, Vq)
        kept.append(Ki * np.sqrt((Vqs[:, trim : - trim] ** 2).mean()))
        dropped.append(Kp * np.sqrt((np.gradient(Vqs, dt, axis=1)[:, trim : - trim] ** 2).mean()))
        noise.append(Ki * np.sqrt(((Vq - Vqs)[:, trim : - trim] ** 2).mean()))

    x = np.arange(W)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.25, kept,    0.25, label="|Ki*Vq|  (the term the code keeps)")
    ax.bar(x,        dropped, 0.25, label="|Kp*dVq/dt|  (the term autograd drops)")
    ax.bar(x + 0.25, noise,   0.25, label="Ki*(Vq sensor noise)  (irreducible)")
    ax.set_yscale("log"); ax.set_xlabel("window index")
    ax.set_ylabel("RMS [rad/s^2]"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("Residual budget: a PERFECT model would still score |Kp*dVq/dt|")
    fig.tight_layout()
    fig.savefig(GRAPHS / "06_residual_budget.png", dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    data, meta, prep, model, ck, tr, va = load_all()
    val_runs = sorted(set(prep["run_id"][va].tolist()))
    print(f"{meta['n_runs']} runs, W={meta['W']}, S={meta['S']}, dt={meta['dt']}")

    fig_initial_conditions(data, meta);                          print("01 done")
    fig_lock_check(data, meta);                                  print("02 done")
    fig_prediction_vs_truth(prep, meta, model, ck, val_runs);    print("03 done")
    fig_window_sweep(prep, meta, model, ck, val_runs);           print("04 done")
    fig_error_by_window(data, prep, meta, model, ck, tr, va);    print("05 done")
    fig_residual_budget(data, meta);                             print("06 done")
    print(f"-> {GRAPHS}")