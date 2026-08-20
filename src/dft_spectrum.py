"""Where is the power? The bandwidth question behind `max_freq`.

    python src/dft_spectrum.py runs/famD_W40_n5000_W40_F4_mf503_wp0.3_s1sp0.pth

F50 changed what this script should ask. `exp8` showed F=1 (the single frequency 503
rad/s) loses to F=4 with no overlap, so **there is no resonance at 80 Hz to hunt for** --
`max_freq` sets the top of a BASIS, not a matched filter. The question is therefore:

    is the thing the trunk has to represent contained below ~503 rad/s?

What the trunk actually has to fit is `target_theta`, the DEVIATION `theta - (theta0 +
w_base*t)` on one window -- not the raw angle, which is dominated by the 50 Hz ramp that
D5 already subtracts. Three spectra, all per-window and averaged over runs:

    target_theta   what the trunk must represent
    Vq             the forcing the physics term sees
    residual       theta_pred - theta_true, i.e. what the model FAILED to represent

The cumulative-power curve is the one that answers the question: read off the fraction of
power below 503 rad/s. If `target_theta` is ~fully contained there and the residual is
not, the basis is the binding constraint and "why 503" is answered.
"""
import argparse

import numpy as np
import torch

from dataset_generator import Dataset_Creator
from paths import GRAPHS, ROOT
from pll_infer import predict_window
from train_pll import group_split, load_checkpoint, prepare

MARKS = [(314.0, r"$\omega_{base}$ = 314"), (503.0, "mf = 503"), (628.0, r"$2\omega$ = 628")]


def spectra(x, dt):
    """x (n, S) real -> (freqs [rad/s], mean power spectrum). Hann window: a window of the
    signal is not periodic, and the leakage from that would smear exactly the low-frequency
    region this script is trying to read."""
    n, S = x.shape
    w = torch.hann_window(S, periodic=False, dtype=x.dtype)
    X = torch.fft.rfft((x - x.mean(dim=1, keepdim=True)) * w, dim=1)
    p = (X.abs() ** 2).mean(dim=0)
    f = torch.fft.rfftfreq(S, d=dt) * 2 * np.pi              # rad/s
    return f.numpy(), p.numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--dataset", default=None, help="default: read from the checkpoint")
    p.add_argument("--n_runs", type=int, default=60)
    a = p.parse_args()

    ck_path = ROOT / a.ckpt if not a.ckpt.startswith("/") else a.ckpt
    model, ck = load_checkpoint(ck_path)
    meta = ck["data_meta"]
    dataset = a.dataset or f"famD_W{meta['W']}.npz"
    data, meta2 = Dataset_Creator.load_dataset(dataset)
    prep = prepare(data)
    W, S, dt = meta2["W"], meta2["S"], meta2["dt"]
    _, va = group_split(prep["run_id"], 0.15, 0)
    val_runs = sorted(set(prep["run_id"][va].tolist()))[:a.n_runs]

    t = ck["t_local"]
    t_ext = torch.cat([t, t[-1:] + float(t[1] - t[0])])
    rows, resid = [], []
    for r in val_runs:
        for k in range(W):
            i = r * W + k
            rows.append(i)
            th, _ = predict_window(model, ck, prep["theta0_abs"][i], prep["omega0"][i],
                                   prep["Va"][i], prep["Vb"][i], prep["Vc"][i], t_ext)
            resid.append(th[:-1] - prep["theta_abs"][i])
    idx = torch.tensor(rows)
    sigs = {"target_theta  (what the trunk must fit)": prep["target_theta"][idx].double(),
            "Vq  (the forcing)": prep["Vq"][idx].double(),
            "residual  (what the model missed)": torch.stack(resid).double()}

    # FULL-RUN spectrum. The per-window transform cannot resolve below the window's own
    # fundamental (503 rad/s at W=40) BY CONSTRUCTION, so on its own it can only say "the
    # power is in the first bin". Stitching each run gives 40x the resolution and is what
    # actually located the content -- see F51.
    from train_pll import OMEGA_BASE
    tt = torch.arange(W * S, dtype=torch.float64) * dt
    full = []
    for r in val_runs:
        th = torch.cat([prep["theta_abs"][r * W + k] for k in range(W)]).double()
        full.append(th - (th[0] + OMEGA_BASE * tt))
    f_full, p_full = spectra(torch.stack(full), dt)
    c_full = np.cumsum(p_full) / p_full.sum()
    print(f"\nFULL-RUN deviation spectrum (resolution {f_full[1]:.1f} rad/s):")
    for lim in [17.3, 50, 126, 251, 503]:
        print(f"   power below {lim:6.1f} rad/s : {100*c_full[np.searchsorted(f_full, lim)-1]:6.2f}%")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.0))
    print(f"{dataset}: {len(val_runs)} runs x {W} windows, S={S}, dt={dt*1e6:.0f} us")
    print(f"window = {S*dt*1e3:.2f} ms -> lowest resolvable frequency "
          f"{2*np.pi/(S*dt):.0f} rad/s, Nyquist {np.pi/dt:.0f} rad/s\n")
    print(f"  {'signal':44s} {'% power below 503':>18s} {'below 628':>11s} {'f_90%':>10s}")
    for (name, x), col in zip(sigs.items(), ["tab:blue", "tab:orange", "tab:red"]):
        f, pw = spectra(x, dt)
        c = np.cumsum(pw) / pw.sum()
        below = lambda lim: 100 * c[np.searchsorted(f, lim) - 1]
        f90 = f[np.searchsorted(c, 0.90)]
        print(f"  {name:44s} {below(503):17.1f}% {below(628):10.1f}% {f90:9.0f}")
        ax[0].loglog(f[1:], pw[1:] / pw[1:].max(), lw=1.2, color=col, label=name)
        ax[1].semilogx(f[1:], c[1:] * 100, lw=1.6, color=col, label=name)

    ax[2].semilogx(f_full[1:], c_full[1:] * 100, lw=1.8, color="tab:purple")
    ax[2].axhline(90, color="gray", ls="--", lw=1)
    ax[2].axvline(17.32, color="tab:green", lw=1.4)
    ax[2].annotate(r"$\omega_n=\sqrt{K_i}$=17.3" + f"\n{100*c_full[np.searchsorted(f_full,17.3)-1]:.0f}% of power below here",
                   (17.32, 45), fontsize=8, color="tab:green", ha="left")
    ax[2].set_ylabel("cumulative power [%]")
    ax[2].set_title("FULL-RUN deviation — 40x the resolution.\nThere is no power at 503 at all",
                    fontsize=10)

    for a_ in ax:
        for fx, lab in MARKS:
            a_.axvline(fx, color="k", ls=":", lw=1.1, alpha=0.7)
            a_.annotate(lab, (fx, a_.get_ylim()[1]), rotation=90, fontsize=7,
                        va="top", ha="right")
        a_.set_xlabel("frequency [rad/s]"); a_.grid(alpha=0.3, which="both")
        a_.legend(fontsize=7.5, loc="lower right")
    ax[0].set_ylabel("power, normalised"); ax[0].set_title("Power spectrum, per window", fontsize=10)
    ax[1].set_ylabel("cumulative power [%]"); ax[1].axhline(90, color="gray", ls="--", lw=1)
    ax[1].set_title("Cumulative power, per window — limited by the 503 rad/s bin", fontsize=10)
    fig.suptitle(f"What the trunk has to represent, and how far up in frequency it lives\n"
                 f"{S*dt*1e3:.1f} ms windows, {len(val_runs)} validation runs", fontsize=10)
    fig.tight_layout()
    GRAPHS.mkdir(exist_ok=True)
    fig.savefig(GRAPHS / "20_dft_spectrum.png", dpi=160)
    print(f"\n-> {GRAPHS / '20_dft_spectrum.png'}")


if __name__ == "__main__":
    main()
