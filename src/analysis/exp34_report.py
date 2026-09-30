"""exp34 -- the hyperparameter re-sweep at the FINAL configuration, on both common tests.

    python src/analysis/exp34_report.py              # score what exists, graphs/33
    python src/analysis/exp34_report.py --plot_only  # redraw from the saved JSON, no rescoring

Every arm is the flagship's recipe (famO40k: limiter, tunable gains, anchor trained in,
L3_w128, F4 mf503, w_phys 0.3, W=40) with ONE knob moved, 8 seeds each. The baseline is
exp33's anchor-trained arm -- the flagship (s6) and its 7 siblings -- not retrained. Design,
verdict rule and the pre-registered predictions: hpc/exp34a_resweep.txt.

THE COMMON TESTS are graphs/30's: famO's and famQ's 150 val runs, both different LHS draws
from famO40k. A result counts only if it holds on BOTH.

OTHER WINDOW LENGTHS on the same runs: a run is 5000 contiguous samples and its 40 test rows
concatenate back into it, so a W=20 (S=250) or W=50 (S=100) model is rolled out on exactly
the same signal, in its own window length. For W=40 models this is round29_31.part1's
rollout op for op, and the baseline's scores are checked against the values exp32_report
saved before anything else is trusted.

Scores only models whose sweep RECORD exists -- a .pth pulled from a job that is still
training may be a mid-training checkpoint. Models already in the saved JSON are not rescored.
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
from scipy.stats import mannwhitneyu

from common_test import load_f32
from dataset_generator import Dataset_Creator
from paths import ROOT, graphs as _graphs
from pll_infer import predict_window
from train_pll import prepare, group_split

GREY, ORANGE, AQUA, BLUE = "#555555", "#eb6834", "#1baf7a", "#2a78d6"
CAP = 1200                       # --epochs of every arm: a record at the cap is an upper bound
ALPHA = 0.05 / 8                 # Bonferroni over the 8 arms, pre-registered
TIE = 1.05                       # a ratio inside 1.05x is a tie whatever p says
SWEEP = ROOT / "Hyperparameter_sweep"


def tag(W=40, F=4, mf="503", wp="0.3", s=0):
    return f"famO40k_W{W}_n40000_W{W}_F{F}_mf{mf}_wp{wp}_s{s}sp0_L3_w128_ao_g"


#        key         label                          knob                    colour  records in
ARMS = [("base",     "baseline\n(flagship recipe)", {},                     GREY,   "sweeps_exp33"),
        ("wp0",      "w_phys 0",                    {"wp": "0"},            BLUE,   "sweeps_exp34"),
        ("wp0.1",    "w_phys 0.1",                  {"wp": "0.1"},          BLUE,   "sweeps_exp34"),
        ("wp1",      "w_phys 1",                    {"wp": "1"},            BLUE,   "sweeps_exp34"),
        ("F8mf1006", "F8 to 1006\n(lowest 126)",    {"F": 8, "mf": "1006"}, ORANGE, "sweeps_exp34"),
        ("F4mf1006", "F4 to 1006\n(lowest 251)",    {"mf": "1006"},         ORANGE, "sweeps_exp34"),
        ("F4mf251",  "F4 to 251\n(lowest 63)",      {"mf": "251"},          ORANGE, "sweeps_exp34"),
        ("W20",      "W=20\n(25 ms windows)",       {"W": 20},              AQUA,   "sweeps_exp34"),
        ("W50",      "W=50\n(10 ms windows)",       {"W": 50},              AQUA,   "sweeps_exp34")]


def records(arm):
    """{tag: sweep record} for the seeds of one arm whose training has FINISHED."""
    _, _, knob, _, d = arm
    out = {}
    for s in range(8):
        p = SWEEP / d / f"{tag(**knob, s=s)}.json"
        if p.exists() and (ROOT / "runs" / f"{tag(**knob, s=s)}.pth").exists():
            out[tag(**knob, s=s)] = json.load(open(p))
    return out


def test_runs(split, n_runs):
    """The split's first n_runs val runs, each concatenated back into one 0.5 s run."""
    data, meta = Dataset_Creator.load_dataset(split)
    torch.set_default_dtype(torch.float32)
    prep, W = prepare(data), meta["W"]
    _, va = group_split(prep["run_id"], 0.15, 0)
    runs = []
    for r in sorted(set(prep["run_id"][va].tolist()))[:n_runs]:
        rows, r0 = slice(r * W, (r + 1) * W), r * W
        flat = {k: prep[k][rows].reshape(-1) for k in ("Va", "Vb", "Vc", "theta_abs", "target_omega")}
        runs.append({**flat, "th0": prep["theta0_abs"][r0], "om0": prep["omega0"][r0],
                     "kp": prep["kp"][r0], "ki": prep["ki"][r0]})
    return runs


def score_model(name, runs):
    """Deployed rollout in the model's OWN window length. Same metrics as round29_31.part1."""
    model, ck = load_f32(ROOT / "runs" / f"{name}.pth")
    S, t = ck["data_meta"]["S"], ck["t_local"]
    t_ext = torch.cat([t, t[-1:] + (t[1] - t[0])])
    th_rms, th_max, om_rms = [], [], []
    with torch.no_grad():
        for d in runs:
            th0, om0, pth, pom = d["th0"], d["om0"], [], []
            for k in range(d["Va"].numel() // S):
                sl = slice(k * S, (k + 1) * S)
                th, om = predict_window(model, ck, th0, om0, d["Va"][sl], d["Vb"][sl], d["Vc"][sl],
                                        t_ext, d["kp"], d["ki"])
                pth.append(th[:-1]); pom.append(om[:-1]); th0, om0 = th[-1], om[-1]   # the handover
            e, eo = torch.cat(pth) - d["theta_abs"], torch.cat(pom) - d["target_omega"]
            th_rms.append(float(e.pow(2).mean().sqrt())); th_max.append(float(e.abs().max()))
            om_rms.append(float(eo.pow(2).mean().sqrt()))
    return {"rms": float(np.mean(th_rms)), "max": float(np.max(th_max)), "om_rms": float(np.mean(om_rms))}


def score(split, n_runs):
    out = SWEEP / f"exp34_{_Path(split).stem}.json"
    have = json.load(open(out)) if out.exists() else {}
    todo = [n for arm in ARMS for n in records(arm) if n not in have]
    print(f"{split}: {len(have)} models already scored, {len(todo)} to score")
    if todo:
        runs = test_runs(split, n_runs)
        for i, n in enumerate(todo, 1):
            have[n] = score_model(n, runs)
            json.dump(have, open(out, "w"), indent=1)          # save as we go
            print(f"  [{i}/{len(todo)}] {n}  {have[n]['rms']:.4e}", flush=True)

    # the harness check: the baseline must reproduce what exp32_report measured on these runs
    ref = [r["rms"] for r in json.load(open(SWEEP / f"exp32_{_Path(split).stem}.json"))["famO40k_ao"]]
    mine = [have[tag(s=s)]["rms"] for s in range(8)]
    worst = max(abs(a - b) / b for a, b in zip(mine, ref))
    if worst > 1e-6:
        raise SystemExit(f"HARNESS MISMATCH on {split}: baseline differs from exp32's saved scores "
                         f"by {worst:.1e} relative -- nothing below would be comparable to graphs/30")
    print(f"  harness check: baseline reproduces exp32's saved scores to {worst:.1e} (relative)")
    return have


def compare(scored, arm):
    """Per split: (n, median, ratio = baseline median / arm median, two-sided MW p,
    P(an arm seed beats a baseline seed))."""
    base = [scored[n]["rms"] for n in records(ARMS[0]) if n in scored]
    v = [scored[n]["rms"] for n in records(arm) if n in scored]
    if not v:
        return None
    ratio = st.median(base) / st.median(v)
    p = 1.0 if arm is ARMS[0] else float(mannwhitneyu(v, base, alternative="two-sided").pvalue)
    sup = float(np.mean([a < b for a in v for b in base]))
    return len(v), st.median(v), ratio, p, sup


def verdict(rows):
    if all(r is None for r in rows):
        return "not trained yet"
    if any(r is None or r[0] < 8 for r in rows):
        return "PRELIMINARY"
    ratios, ps = [r[2] for r in rows], [r[3] for r in rows]
    if all(q >= TIE for q in ratios) and all(p < ALPHA for p in ps):
        return "BETTER -> CHANGES THE RECIPE" if all(q >= 1.10 for q in ratios) else "BETTER (< 1.10x: flagship stays)"
    if all(q <= 1 / TIE for q in ratios) and all(p < ALPHA for p in ps):
        return "WORSE"
    return "tie"


def table(refs):
    print(f"\n{'arm':26s}" + "".join(f"{_Path(sp).stem + ': median  x base  p       P(win)':>44s}" for sp, _ in refs)
          + "   epochs     own-val rms   verdict")
    for arm in ARMS:
        rows = [compare(ref, arm) for _, ref in refs]
        rec = records(arm)
        ep = [r["epochs_run"] for r in rec.values()]
        capped = sum(e >= CAP for e in ep)
        cells = "".join(f"{'-- not scored yet':>44s}" if r is None else
                        f"{'n=%d' % r[0]:>8s} {r[1]:.3e} {r[2]:6.2f}x" +
                        (f"{'--':>8s} {'--':>6s}" if arm is ARMS[0] else f" {r[3]:7.1e} {r[4]:6.2f}")
                        for r in rows)
        own = f"{st.median(r['rollout_full_rms'] for r in rec.values()):.3e}" if rec else "--"
        v = "baseline" if arm is ARMS[0] else verdict(rows)
        if capped:
            v += f"  !! CAPPED {capped}/{len(ep)}: upper bound"
        print(f"{arm[1].replace(chr(10), ' '):26s}{cells}   "
              f"{(f'{min(ep)}-{max(ep)}' if ep else '--'):>9s}   {own:>9s}   {v}")
    print(f"\nratio = baseline median / arm median (>1 = the arm is better). p = two-sided Mann-Whitney, "
          f"8 vs 8; the rule needs p < {ALPHA:.4f} on BOTH tests.\nP(win) = share of (arm seed, baseline seed) "
          f"pairs where the arm seed is better. own-val = famO40k's own 150 val runs from the sweep "
          f"record: the same runs for every arm, but also the early-stopping set.")


def panel(ax, ref, split, n_runs):
    rng = np.random.default_rng(0)
    base = compare(ref, ARMS[0])[1]
    b = [ref[n]["rms"] for n in records(ARMS[0])]
    ax.axhspan(min(b), max(b), color=GREY, alpha=.08, lw=0)
    ticks, labels = [], []
    for i, arm in enumerate(ARMS):
        key, lbl, _, c, _ = arm
        r = compare(ref, arm)
        ticks.append(i); labels.append(lbl + (f"\nn={r[0]}" if r else "\nrunning"))
        if r is None:
            ax.annotate("still\nrunning", (i, base * 0.8), ha="center", fontsize=8.5, color=GREY)
            continue
        v = [ref[n]["rms"] for n in records(arm) if n in ref]
        ax.scatter(i + rng.uniform(-.13, .13, len(v)), v, s=52, color=c, edgecolor="white",
                   linewidth=1.1, zorder=3)
        ax.hlines(r[1], i - .3, i + .3, color="#222222", lw=2, zorder=4)
        txt = f"{r[2]:.2f}x" + ("" if arm is ARMS[0] else f"\np={r[3]:.0e}")
        ax.annotate(txt, (i, max(v)), xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=8.5, color="#222222")
    for x in (0.5, 3.5, 6.5):                         # baseline | w_phys | comb | window
        ax.axvline(x, color=GREY, lw=.6, alpha=.4)
    ax.set_yscale("log"); ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=8.6)
    ax.set_xlim(-.6, len(ARMS) - .4)
    ax.set_ylabel(f"deployed theta RMS on {_Path(split).stem}'s {n_runs} val runs [rad]", fontsize=9.5)
    ax.grid(alpha=.25, axis="y", which="both")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"test runs from {_Path(split).stem}", fontsize=10)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", nargs="+", default=["famO_W40.npz", "famQ_W40.npz"])
    p.add_argument("--n_runs", type=int, default=150)
    p.add_argument("--out", default="33_exp34_resweep.png")
    p.add_argument("--plot_only", action="store_true",
                   help="redraw from the saved Hyperparameter_sweep/exp34_<split>.json, no rescoring")
    a = p.parse_args()

    load = lambda sp: json.load(open(SWEEP / f"exp34_{_Path(sp).stem}.json"))
    refs = [(sp, load(sp) if a.plot_only else score(sp, a.n_runs)) for sp in a.splits]
    table(refs)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(refs), figsize=(12 * len(refs), 6.2), squeeze=False)
    for ax, (sp, ref) in zip(axes[0], refs):
        panel(ax, ref, sp, a.n_runs)
    fig.suptitle("exp34: the flagship's recipe with one knob moved, 8 seeds per arm "
                 "(grey = baseline = exp33's anchor-trained arm, flagship s6 among them)\n"
                 "bar = median over seeds; number = baseline median / arm median (>1 = the arm is better), "
                 f"p = two-sided Mann-Whitney vs the baseline. An arm counts only at p < {ALPHA:.4f} on BOTH panels",
                 fontsize=11)
    fig.tight_layout()
    out = _graphs(_Path(a.out)); out.parent.mkdir(exist_ok=True); fig.savefig(out, dpi=150)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
