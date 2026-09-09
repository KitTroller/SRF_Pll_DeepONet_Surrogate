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
    dict(name="limiter + TUNABLE gains", free="famU_W40", noisy="famO_W40", limit=LIMIT),
    dict(name="no limiter, fixed gains", free="famV_W40", noisy="famR_W40", limit=None),
)


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
    """Per-run theta RMS over the full 0.5 s recurrent rollout. Kept run-wise so the
    spread is visible -- a 4-seed median with no spread is how the hidden-dim figure
    misled once."""
    torch.set_default_dtype(torch.float32)
    out = []
    for r in range(th_t.shape[0]):
        pth, _ = rollout(model, ck, (V[0][r], V[1][r], V[2][r]),
                         th_t[r, 0].float(), om_t[r, 0].float(), KP, KI)
        out.append(float((pth.double() - th_t[r]).pow(2).mean().sqrt()))
    torch.set_default_dtype(torch.float64)
    return np.array(out)


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

    cells = {}                                  # (pair, model_noise, truth_noise) -> values
    for pr in PAIRS:
        for tn in (False, True):                # truth: noise-free, then noisy
            V0, V1, V2, th_t, om_t = make_truth(pr["limit"], tn, a.n_runs, a.seed)
            for mn, fam in ((False, pr["free"]), (True, pr["noisy"])):
                acc = []
                for q in checkpoints(fam):
                    m, ck = load_f32(q)
                    acc.append(per_run_rms(m, ck, (V0, V1, V2), th_t, om_t))
                cells[(pr["name"], mn, tn)] = np.concatenate(acc)
                print(f"  {pr['name'][:22]:22s} model={'noisy' if mn else 'free ':5s} "
                      f"truth={'noisy' if tn else 'free ':5s} "
                      f"median={np.median(cells[(pr['name'], mn, tn)]):.4e}  "
                      f"({len(checkpoints(fam))} seeds x {a.n_runs} runs)")
    PS.pll_constants.freq_limit = None
    PS.initial_conditions_config.white_noise_on_flag = True

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig, ax = plt.subplots(1, 4, figsize=(24, 5.4),
                           gridspec_kw=dict(width_ratios=[1, 1, 1.55, 1.15]))

    for i, pr in enumerate(PAIRS):
        M = np.array([[np.median(cells[(pr["name"], mn, tn)]) for tn in (False, True)]
                      for mn in (False, True)])
        base = M[1, 1]                          # noisy model on noisy truth = what we ship
        # colour the RATIO, on one shared scale, or the two panels cannot be read against
        # each other -- an independent vmin/vmax per panel makes 0.89x and 0.49x the same
        # shade of green and 14x and 39x the same red
        ax[i].imshow(M / base, cmap="RdYlGn_r", norm=LogNorm(vmin=0.4, vmax=40))
        for r in range(2):
            for c in range(2):
                ax[i].text(c, r, f"{M[r, c]:.3e}\n{M[r, c]/base:.2f}x", ha="center",
                           va="center", fontsize=13,
                           fontweight="bold" if (r, c) == (1, 1) else "normal")
        ax[i].set_xticks([0, 1]); ax[i].set_yticks([0, 1])
        ax[i].set_xticklabels(["noise-FREE", "NOISY"], fontsize=10)
        ax[i].set_yticklabels([f"noise-free\n({pr['free'].split('_')[0]})",
                               f"noisy\n({pr['noisy'].split('_')[0]})"], fontsize=10)
        ax[i].set_xlabel("truth it is scored on"); ax[i].set_ylabel("what it was trained on")
        ax[i].set_title(f"{pr['name']}\n$\\times$ = relative to what we ship (bold)",
                        fontsize=10.5)

    # every run behind every median, so a 4-seed cell cannot pass for a distribution
    rng = np.random.default_rng(0)
    xs, labs, cols = [], [], []
    for i, pr in enumerate(PAIRS):
        for mn in (False, True):
            for tn in (False, True):
                x = len(xs) + 0.6 * i
                v = cells[(pr["name"], mn, tn)]
                ax[2].scatter(np.full(len(v), x) + rng.uniform(-.12, .12, len(v)), v,
                              s=9, alpha=.30, color="tab:red" if mn else "tab:blue")
                ax[2].hlines(np.median(v), x - .3, x + .3,
                             color="tab:red" if mn else "tab:blue", lw=3, zorder=4)
                xs.append(x)
                labs.append(f"{'noisy' if mn else 'free'} model\non {'noisy' if tn else 'free'}")
    ax[2].set_xticks(xs); ax[2].set_xticklabels(labs, fontsize=7.5)
    ax[2].axvline(np.mean(xs[3:5]), color="k", lw=1, ls="--")
    ax[2].text(.25, .97, PAIRS[0]["name"], transform=ax[2].transAxes, ha="center",
               va="top", fontsize=9.5)
    ax[2].text(.76, .97, PAIRS[1]["name"], transform=ax[2].transAxes, ha="center",
               va="top", fontsize=9.5)
    ax[2].set_yscale("log"); ax[2].grid(alpha=.3, which="both")
    ax[2].set_ylabel("per-run $\\theta$ RMS over 0.5 s [rad]")
    ax[2].set_title("Every run behind every median.  bar = median", fontsize=10.5)

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
    ax[3].bar(x, r1s, .72, color=cols, edgecolor="k", lw=.5)
    ax[3].set_yscale("log"); ax[3].set_xticks(x); ax[3].set_xticklabels(labs, fontsize=8.5)
    ax[3].set_ylabel("median physics residual $r_1$", color="k")
    ax[3].grid(alpha=.3, axis="y", which="both")
    tw = ax[3].twinx()
    tw.plot(x, gaps, "k^", ms=9)               # no connecting line: these are four
                                               # separate families, not a trend in x
    tw.set_ylabel("val / train loss  (overfit gap)", fontsize=9.5)
    tw.set_ylim(0, max(gaps) * 1.25)
    ax[3].set_title("The residual floor IS real (bars, 6-30x lower).\nIt just does not "
                    "reach deployed error — and the\noverfit gap (triangles) widens "
                    "without noise.", fontsize=10)

    fig.suptitle("Removing the white measurement noise buys almost nothing deployed, and "
                 "costs 14-39x if any noise is present.\nAll four cells are the SAME "
                 "freshly generated trajectories at (Kp,Ki)=(25,300) — never a per-family "
                 "validation split.", fontsize=12)
    fig.tight_layout()
    out = _graphs(a.out); fig.savefig(out, dpi=140)
    print(f"\n-> {out}")

    for pr in PAIRS:
        base = np.median(cells[(pr["name"], True, True)])
        print(f"\n{pr['name']}   (x relative to the noisy model on noisy truth)")
        print(f"  {'':22s}{'truth: FREE':>22s}{'truth: NOISY':>22s}")
        for mn in (False, True):
            row = ""
            for tn in (False, True):
                m = np.median(cells[(pr["name"], mn, tn)])
                row += f"{m:13.3e}{m/base:8.2f}x"
            print(f"  model {'NOISY' if mn else 'FREE ':5s}          {row}")


if __name__ == "__main__":
    main()
