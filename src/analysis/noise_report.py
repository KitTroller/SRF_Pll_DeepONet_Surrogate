"""Does removing the white measurement noise lower the error floor? graphs/27.

    python src/noise_report.py              # ~15 min

The supervisor's EMT simulation carries no white measurement noise. Ours adds
0.1*(rand-0.5) pu to Va/Vb/Vc, and F48/F49 established that this noise -- not the
integrator -- sets the accuracy floor. famU/famV are famO/famR regenerated with
`--no_white_noise` at the SAME --lhs_seed, and because `noise` is drawn BEFORE the flag
is tested in `_grid_phases`, the RNG stream is identical: same ICs, same harmonics, same
faults, same gain draws. Only the noise term differs.

WHY THIS IS A 2x2 AND NOT TWO NUMBERS. The obvious experiment -- train without noise,
see if the number drops -- measures the model on its own noise-free validation split and
compares it to famO's noisy one. That is F59/F61 exactly: two different physics, two
different difficulties, and the difference gets read as a model difference. Worse, it is
also F62: removing FAULTS from training bought nothing on clean data and cost 5-9x on
sags, because the model had never seen that region of input space. A noise-free model has
never seen noisy Va/Vb/Vc at all.

So every model is scored on BOTH truths, on identical freshly generated trajectories:

                      truth: noise-free      truth: noisy
    model: noise-free      the new floor     the extrapolation cost
    model: noisy           does noise-       THE STATUS QUO -- what we
                           training hurt?    ship today, the denominator

Both pairs are evaluated at the nominal (Kp, Ki) = (25, 300) so the limited and unlimited
rows differ ONLY by the limiter, not by the operating point.
"""

# src/ on the path: these scripts live in src/analysis/ but import the pipeline
# modules (paths, PLL_Simulator, train_pll, sweep) that stay in src/. Running
# `python src/analysis/foo.py` puts src/analysis on sys.path, not src/.
# Same pattern as hpc/generate_family.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import argparse
import glob

import numpy as np
import torch

import PLL_Simulator as PS
from common_test import load_f32, rollout, truth
from paths import graphs as _graphs

KP, KI = 25.0, 300.0            # config nominal; famV/famR are fixed-gain models AT this
LIMIT = 18.8496                 # 2*pi*3 rad/s

PAIRS = (
    dict(name="limiter + FIXED gains", free="famY_W40", noisy="famN_W40", limit=LIMIT),
    dict(name="limiter + TUNABLE gains", free="famU_W40", noisy="famO_W40", limit=LIMIT),
    dict(name="NO limiter + FIXED gains", free="famV_W40", noisy="famR_W40", limit=None),
    dict(name="NO limiter + TUNABLE gains", free="famW_W40", noisy="famX_W40", limit=None),
)
# THE COMPLETE 2x2x2 (limiter x gains x noise), in that order so the panels read as a
# factorial. exp23 gave corners 2 and 3 and moved two factors at once, so its 1.02x vs
# 1.59x could not be attributed (F68). exp24 added corner 4, isolating the limiter at
# tunable gains. exp27's famY is corner 1 and closes it: pairs 1-2 differ only by the
# gains with the limiter on, 3-4 only by the gains with it off, 1-3 and 2-4 only by the
# limiter. Every pair shares its partner's --lhs_seed, so all eight families are bit-
# paired with their twin and differ by the noise term alone.


def make_truth(limit, white_noise, n_runs, seed):
    """Ground truth at one (limiter, noise) setting. Both flags are read in
    `PLLSimulator.__init__`, so they must be patched BEFORE `truth()` builds the sim --
    and on PLL_Simulator's own module objects, because `Dataset_Creator` and the
    simulator hold separate OmegaConf loads. This is the trap that has bitten three
    times; see the note in hpc/generate_family.py."""
    PS.pll_constants.freq_limit = limit
    PS.initial_conditions_config.white_noise_on_flag = white_noise
    return truth(KP, KI, n_runs, seed=seed)


def per_run_rms(model, ck, V, th_t, om_t):
    """Per-run DEPLOYED theta error over the full 0.5 s: RMS and worst case.

    Deployed means recurrent -- `rollout` hands the model's own final state to the next
    window, 40 times, with no ground truth injected. That is the same quantity the sweep
    records call `rollout_full_rms`, not the teacher-forced `per_window_rms`.

    The worst case is returned alongside because RMS averages over 5000 samples and a
    limiter-compliance claim lives or dies on the peak. Kept run-wise so the spread is
    visible -- a 4-seed median with no spread is how the hidden-dim figure misled once."""
    torch.set_default_dtype(torch.float32)
    rms, mx = [], []
    for r in range(th_t.shape[0]):
        pth, _ = rollout(model, ck, (V[0][r], V[1][r], V[2][r]),
                         th_t[r, 0].float(), om_t[r, 0].float(), KP, KI)
        e = (pth.double() - th_t[r]).abs()
        rms.append(float(e.pow(2).mean().sqrt())); mx.append(float(e.max()))
    torch.set_default_dtype(torch.float64)
    return np.array(rms), np.array(mx)


def checkpoints(fam):
    """The plain L2_w64 seeds only. `_L` tags are the capacity grid (F66) and would mix
    architectures into a comparison whose only variable is meant to be the noise."""
    q = sorted(p for p in glob.glob(f"runs/{fam}_*sp0*.pth") if "_L" not in p)
    if not q:
        raise SystemExit(f"no plain checkpoints for {fam}")
    return q


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="27_white_noise.png")
    p.add_argument("--n_runs", type=int, default=12)
    p.add_argument("--seed", type=int, default=0, help="the trajectory draw, not a model seed")
    a = p.parse_args()

    cells, peaks = {}, {}                       # (pair, model_noise, truth_noise) -> values
    for pr in PAIRS:
        for tn in (False, True):                # truth: noise-free, then noisy
            V0, V1, V2, th_t, om_t = make_truth(pr["limit"], tn, a.n_runs, a.seed)
            for mn, fam in ((False, pr["free"]), (True, pr["noisy"])):
                acc, acx = [], []
                for q in checkpoints(fam):
                    m, ck = load_f32(q)
                    r_, x_ = per_run_rms(m, ck, (V0, V1, V2), th_t, om_t)
                    acc.append(r_); acx.append(x_)
                cells[(pr["name"], mn, tn)] = np.concatenate(acc)
                peaks[(pr["name"], mn, tn)] = np.concatenate(acx)
                print(f"  {pr['name'][:22]:22s} model={'noisy' if mn else 'free ':5s} "
                      f"truth={'noisy' if tn else 'free ':5s} "
                      f"median={np.median(cells[(pr['name'], mn, tn)]):.4e}  "
                      f"worst={peaks[(pr['name'], mn, tn)].max():.4e}  "
                      f"({len(checkpoints(fam))} seeds x {a.n_runs} runs)")
    PS.pll_constants.freq_limit = None
    PS.initial_conditions_config.white_noise_on_flag = True

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    # Two rows once there are four pairs: one row of four matrices plus the scatter and
    # the mechanism panel would be 34 inches wide and unreadable at any sane dpi.
    npair = len(PAIRS)
    fig = plt.figure(figsize=(5.3 * npair, 10.6))
    gs = fig.add_gridspec(2, npair, height_ratios=[1, 1.05], hspace=.28)
    ax = [fig.add_subplot(gs[0, i]) for i in range(npair)]
    ax_sc = fig.add_subplot(gs[1, :npair - 1])
    ax_mech = fig.add_subplot(gs[1, npair - 1])

    for i, pr in enumerate(PAIRS):
        M = np.array([[np.median(cells[(pr["name"], mn, tn)]) for tn in (False, True)]
                      for mn in (False, True)])
        base = M[1, 1]                          # noisy model on noisy truth = what we ship
        # colour the RATIO, on one shared scale, or the two panels cannot be read against
        # each other -- an independent vmin/vmax per panel makes 0.89x and 0.49x the same
        # shade of green and 14x and 39x the same red
        ax[i].imshow(M / base, cmap="RdYlGn_r", norm=LogNorm(vmin=0.4, vmax=40))
        P = np.array([[peaks[(pr["name"], mn, tn)].max() for tn in (False, True)]
                      for mn in (False, True)])
        for r in range(2):
            for c in range(2):
                # RMS on top (what the sweep records call rollout_full_rms), then the
                # ratio, then the WORST single sample across every seed and run. RMS
                # averages 5000 samples per run; a compliance claim lives on the peak.
                # explicit offsets, not stacked newlines: the two blocks are different
                # sizes and a bbox on the lower one covered the ratio when they shared a
                # centre. Cells span +/-0.5, so -0.13 / +0.24 stay well inside.
                ax[i].text(c, r - 0.13, f"{M[r, c]:.3e}\n{M[r, c]/base:.2f}x",
                           ha="center", va="center", fontsize=12.5,
                           fontweight="bold" if (r, c) == (1, 1) else "normal")
                ax[i].text(c, r + 0.24, f"worst {P[r, c]:.2e}", ha="center", va="center",
                           fontsize=8.5, color="#222222",
                           # the cells span green to dark red; grey-on-red is unreadable
                           bbox=dict(fc="white", alpha=.72, ec="none", pad=1.5))
        ax[i].set_xticks([0, 1]); ax[i].set_yticks([0, 1])
        ax[i].set_xticklabels(["noise-FREE", "NOISY"], fontsize=10)
        ax[i].set_yticklabels([f"noise-free\n({pr['free'].split('_')[0]})",
                               f"noisy\n({pr['noisy'].split('_')[0]})"], fontsize=10)
        ax[i].set_xlabel("truth it is scored on")
        if i == 0:      # only the leftmost: four copies get squeezed between the panels
            ax[i].set_ylabel("what it was trained on")
        ax[i].set_title(f"{pr['name']}\n$\\times$ = relative to what we ship (bold)",
                        fontsize=10.5)

    # every run behind every median, so a 4-seed cell cannot pass for a distribution
    rng = np.random.default_rng(0)
    xs, labs, bounds = [], [], []
    for i, pr in enumerate(PAIRS):
        first = len(xs)
        for mn in (False, True):
            for tn in (False, True):
                x = len(xs) + 0.7 * i
                v = cells[(pr["name"], mn, tn)]
                ax_sc.scatter(np.full(len(v), x) + rng.uniform(-.12, .12, len(v)), v,
                              s=7, alpha=.28, color="tab:red" if mn else "tab:blue")
                ax_sc.hlines(np.median(v), x - .3, x + .3,
                             color="tab:red" if mn else "tab:blue", lw=3, zorder=4)
                xs.append(x)
                labs.append(f"{'noisy' if mn else 'free'}\non {'noisy' if tn else 'free'}")
        bounds.append((xs[first], xs[-1]))
    for lo, hi in bounds[:-1]:                 # divider between adjacent pair blocks
        ax_sc.axvline(hi + 0.35, color="k", lw=1, ls="--")
    for (lo, hi), pr in zip(bounds, PAIRS):
        ax_sc.text((lo + hi) / 2, .985, pr["name"], transform=ax_sc.get_xaxis_transform(),
                   ha="center", va="top", fontsize=8.5)
    ax_sc.set_xticks(xs); ax_sc.set_xticklabels(labs, fontsize=7)
    ax_sc.set_yscale("log"); ax_sc.grid(alpha=.3, which="both")
    ax_sc.set_ylabel("per-run $\\theta$ RMS over 0.5 s [rad]")
    ax_sc.set_title("Every run behind every median.  bar = median", fontsize=10.5)

    # WHY. The original hypothesis was that the noise sets a floor on the physics
    # residual. It does -- and that is exactly the point: the residual improves 6-30x
    # while the deployed error does not move. These come straight from the sweep records,
    # so this panel costs no extra compute.
    import json
    import os
    def rec(fam):
        d = f"Hyperparameter_sweep/sweeps_{fam}"
        return [json.load(open(os.path.join(d, f))) for f in sorted(os.listdir(d))
                if f.endswith(".json")]
    x, r1s, gaps, cols, labs = [], [], [], [], []
    for pr in PAIRS:
        for mn, fam in ((False, pr["free"]), (True, pr["noisy"])):
            rs = [r for r in rec(fam) if not r.get("n_layers") and not r.get("width")]
            x.append(len(x) + 0.5 * PAIRS.index(pr))
            r1s.append(np.median([r["r1"] for r in rs]))
            gaps.append(np.median([r["val_th"] / r["train_th"] for r in rs]))
            cols.append("tab:red" if mn else "tab:blue")
            labs.append(f"{fam.split('_')[0]}\n{'noisy' if mn else 'free'}")
    ax_mech.bar(x, r1s, .72, color=cols, edgecolor="k", lw=.5)
    ax_mech.set_yscale("log"); ax_mech.set_xticks(x)
    ax_mech.set_xticklabels(labs, fontsize=8)
    ax_mech.set_ylabel("median physics residual $r_1$", color="k")
    ax_mech.grid(alpha=.3, axis="y", which="both")
    tw = ax_mech.twinx()
    tw.plot(x, gaps, "k^", ms=9)               # no connecting line: these are separate
                                               # families, not a trend in x
    tw.set_ylabel("val / train loss  (overfit gap)", fontsize=9.5)
    tw.set_ylim(0, max(gaps) * 1.25)
    ax_mech.set_title("The residual floor IS real (bars).\nIt just does not reach deployed "
                      "error — and the\noverfit gap (triangles) widens without noise.",
                      fontsize=10)

    # Built from the data, not typed. The headline has had to be corrected twice as pairs
    # were added; a computed one cannot go stale.
    def effect(pr):
        """>1 = removing the noise HELPED, on the clean truth, like for like."""
        return (np.median(cells[(pr["name"], True, False)])
                / np.median(cells[(pr["name"], False, False)]))
    worst = max(np.median(cells[(pr["name"], False, True)])
                / np.median(cells[(pr["name"], True, True)]) for pr in PAIRS)
    line = "   ".join(f"{pr['name']}: {effect(pr):.2f}x" for pr in PAIRS)
    fig.suptitle(
        "Removing the white measurement noise, across the complete "
        "(limiter $\\times$ gains) factorial.  >1 = it helped.\n" + line +
        f"\nAnd on noisy truth it costs up to {worst:.0f}x. Every cell is the SAME freshly "
        "generated trajectories at (Kp,Ki)=(25,300) — never a per-family validation split.",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    out = _graphs(a.out); fig.savefig(out, dpi=140)
    print(f"\n-> {out}")

    for pr in PAIRS:
        base = np.median(cells[(pr["name"], True, True)])
        print(f"\n{pr['name']}   (x relative to the noisy model on noisy truth)")
        print(f"  {'':22s}{'truth: FREE':>30s}{'truth: NOISY':>30s}")
        for mn in (False, True):
            row = ""
            for tn in (False, True):
                m = np.median(cells[(pr["name"], mn, tn)])
                row += f"{m:13.3e}{m/base:7.2f}x  peak{peaks[(pr['name'], mn, tn)].max():9.2e}"
            print(f"  model {'NOISY' if mn else 'FREE ':5s}   {row}")


if __name__ == "__main__":
    main()
