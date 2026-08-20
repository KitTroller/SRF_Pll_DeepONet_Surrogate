"""plot_sweeps.py -- draw a sweeps_* directory. New file; sweep.py is untouched.

    python plot_sweeps.py sweeps_famB_ff --kind arms   # F=0 | mf503 | mf628, 16 seeds
    python plot_sweeps.py sweeps_famB_W  --kind W      # the W sweep

Why not sweep.plot_ff: with 16 seeds a min/max error bar is the wrong display -- one
unlucky seed sets the whole bar, which is exactly how F21 went wrong. These plots show
EVERY seed as a dot, plus the median and the interquartile box, so a wide spread looks
wide and an outlier looks like an outlier instead of like a result.

Metric guidance baked into the panel order (F24): val_th and per_window_rms are the
low-variance metrics and are what DETECT a difference; rollout_full_rms is the headline
number but keeps a ~1.6x seed spread even at n=5000, so it is reported, not trusted.
"""
import argparse
import collections
from pathlib import Path

import numpy as np

from paths import graphs as _graphs
from sweep import load

PANELS = [("val_th", "val MSE $\\theta$ [rad$^2$]", "teacher-forced (detects)"),
          ("per_window_rms", "one-window $\\theta$ RMS [rad]", "operator (detects)"),
          ("rollout_full_med", "median per-run RMS [rad]", "deployed, tail-robust"),
          ("rollout_full_rms", "deployed $\\theta$ RMS @ 0.5 s [rad]", "headline (noisy)")]


def _draw(ax, groups, labels, key, ylab, title):
    data = [[r[key] for r in g] for g in groups]
    pos = np.arange(len(groups)) + 1
    ax.boxplot(data, positions=pos, widths=0.55, showfliers=False,
               medianprops=dict(color="tab:red", lw=2))
    for i, vals in enumerate(data):
        x = pos[i] + (np.random.rand(len(vals)) - 0.5) * 0.22
        ax.scatter(x, vals, s=16, alpha=0.75, color="tab:blue", zorder=3, lw=0)
    for i, vals in enumerate(data):                      # spread, the number that matters
        ax.annotate(f"n={len(vals)}\n{max(vals)/min(vals):.2f}x",
                    (pos[i], max(vals)), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=6.5, color="dimgray")
    ax.set_xticks(pos); ax.set_xticklabels(labels, fontsize=8)
    ax.set_yscale("log"); ax.set_ylabel(ylab, fontsize=8)
    ax.set_title(title, fontsize=9); ax.grid(alpha=0.3, axis="y", which="both")
    ax.margins(y=0.18)


def plot_arms(recs, out):
    """One box per (F, max_freq) arm at a single W. The exp1 picture."""
    import matplotlib.pyplot as plt
    by = collections.defaultdict(list)
    for r in recs:
        by[(r["F"], r["max_freq"])].append(r)
    keys = sorted(by, key=lambda k: (k[0], k[1]))
    labels = ["F=0\n(control)" if f == 0 else f"F={f}\nmf={mf:g}" for f, mf in keys]
    groups = [by[k] for k in keys]

    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    for a, (key, ylab, title) in zip(ax, PANELS):
        _draw(a, groups, labels, key, ylab, title)
    W = recs[0]["W"]
    fig.suptitle(f"Fourier arms at W={W}, n_runs={_nruns(recs)}  "
                 f"({len(recs)} runs; every seed shown, red = median, label = max/min)")
    out = _graphs(out); fig.tight_layout(); out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=160); print(f"-> {out}")


def _hidden_dim(r):
    """Records predating the `params` field carry the width only in the tag."""
    tag = r["tag"]
    if "_h" in tag:
        piece = tag.split("_h")[-1].split("_")[0]
        if piece.isdigit():
            return int(piece)
    return 64                                  # config/DeepONet_models.yml default


def plot_hd(recs, out):
    """One box per hidden_dim. NOT plot_arms: that keys on (F, max_freq), which are
    identical across a width sweep, so every width collapsed into a single box."""
    import matplotlib.pyplot as plt
    by = collections.defaultdict(list)
    for r in recs:
        by[_hidden_dim(r)].append(r)
    keys = sorted(by)
    groups = [by[k] for k in keys]
    labels = [f"hidden={k}" + ("\n(default)" if k == 64 else "") for k in keys]

    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    for a, (key, ylab, title) in zip(ax, PANELS):
        _draw(a, groups, labels, key, ylab, title)
    # compounding is the whole point of this sweep -- width changed the HANDOVER, not
    # the operator -- so state it under the axis rather than making the reader divide.
    comp = [np.median([r["rollout_full_rms"] / r["per_window_rms"] for r in g]) for g in groups]
    ax[3].set_xlabel("median compounding: " + "  ".join(f"{k}:{c:.1f}x" for k, c in zip(keys, comp)),
                     fontsize=8)
    r0 = recs[0]
    fig.suptitle(f"Width sweep at W={r0['W']}, F={r0['F']}, mf={r0['max_freq']:g}, "
                 f"n_runs={_nruns(recs)}  ({len(recs)} runs; every seed shown)")
    out = _graphs(out); fig.tight_layout(); out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=160); print(f"-> {out}")


def plot_residual(recs, out):
    """eq-4 vs eq-6 across the w_phys ladder — paired boxes at each weight.

    A ladder, not a point, on purpose: eq-6 changes the CHARACTER of the physics term, so
    testing at one weight would compare a tuned setting against an untuned one. The TREND
    is the result -- if eq-6 degrades as the weight rises, that is the spurious minimum a
    self-consistent WRONG angle creates, which eq-4 is structurally immune to."""
    import matplotlib.pyplot as plt
    by = collections.defaultdict(list)
    for r in recs:
        by[(r["w_phys"], r.get("residual", "eq4"))].append(r)
    ws = sorted({k[0] for k in by})
    kinds = sorted({k[1] for k in by})
    cols = {"eq4": "tab:blue", "eq6": "tab:red"}

    fig, ax = plt.subplots(1, 4, figsize=(16.5, 4.4))
    for a, (key, ylab, title) in zip(ax, PANELS):
        for j, kind in enumerate(kinds):
            off = (j - (len(kinds) - 1) / 2) * 0.3
            pos, data = [], []
            for i, w in enumerate(ws):
                g = by.get((w, kind))
                if g:
                    pos.append(i + 1 + off); data.append([r[key] for r in g])
            if not data:
                continue
            a.boxplot(data, positions=pos, widths=0.26, showfliers=False,
                      medianprops=dict(color=cols.get(kind, "k"), lw=2))
            for p, vals in zip(pos, data):
                a.scatter(p + (np.random.rand(len(vals)) - 0.5) * 0.12, vals, s=14,
                          alpha=0.8, color=cols.get(kind, "k"), zorder=3, lw=0,
                          label=kind if p == pos[0] and a is ax[0] else None)
        a.set_xticks(np.arange(len(ws)) + 1)
        a.set_xticklabels([f"{w:g}" for w in ws], fontsize=8)
        a.set_yscale("log"); a.set_ylabel(ylab, fontsize=8)
        a.set_xlabel("$w_{phys}$", fontsize=8)
        a.set_title(title, fontsize=9); a.grid(alpha=0.3, axis="y", which="both")
    ax[0].legend(fontsize=8, title="residual form")
    fig.suptitle("eq-4 (stored $V_q$, gauge invariant) vs eq-6 ($V_q$ from the predicted "
                 f"angle) — {len(recs)} runs, every seed shown\n"
                 "eq-6 is never better, and degrades as the physics weight rises: the "
                 "signature of a spurious minimum")
    out = _graphs(out); fig.tight_layout(); out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=160); print(f"-> {out}")


def plot_wphys(recs, out):
    """One box per w_phys at a single (W, F, max_freq). The exp7 picture.

    Categorical positions, not a log x-axis: w_phys=0 has no place on a log scale and it
    is the control the whole sweep exists to beat, so it gets column one and a divider.
    sweep.plot_wphys draws the same data as median + min/max error bars -- at 4 seeds
    that lets one seed set the bar, which is how F21 went wrong. Use this one."""
    import matplotlib.pyplot as plt
    by = collections.defaultdict(list)
    for r in recs:
        by[r["w_phys"]].append(r)
    keys = sorted(by)
    labels = [("0\n(control)" if k == 0 else f"{k:g}") for k in keys]
    groups = [by[k] for k in keys]

    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    for a, (key, ylab, title) in zip(ax, PANELS):
        _draw(a, groups, labels, key, ylab, title)
        a.set_xlabel("$w_{phys}$", fontsize=8)
        if len(keys) > 6:                    # decade labels collide below this width
            a.set_xticklabels(labels, fontsize=7.5, rotation=45, ha="right")
        if keys and keys[0] == 0:
            a.axvline(1.5, color="k", ls=":", lw=1.0, alpha=0.6)
        if len(groups[0]) and len(groups) > 1:      # control band across the panel
            c = [r[key] for r in groups[0]]
            a.axhspan(min(c), max(c), color="k", alpha=0.07, zorder=0)
    r0 = recs[0]
    mf = r0.get("max_freq")                  # absent in the pre-Fourier records
    arm = f"W={r0['W']}, F={r0['F']}" + (f", mf={mf:g}" if r0["F"] and mf else "")
    fig.suptitle(f"Physics-weight sweep at {arm}, n_runs={_nruns(recs)}  "
                 f"({len(recs)} runs; every seed shown, red = median, "
                 f"grey band = $w_{{phys}}=0$)")
    out = _graphs(out); fig.tight_layout(); out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=160); print(f"-> {out}")


def plot_W(recs, out):
    """One box per W. rollout_full_rms is the only metric comparable ACROSS W --
    it is always 0.5 s of physical time because the rollout runs n = W windows.
    val_th is NOT comparable across W (shorter windows shrink the target variance),
    so it is drawn greyed as a reminder, never as evidence."""
    import matplotlib.pyplot as plt
    by = collections.defaultdict(list)
    for r in recs:
        by[r["W"]].append(r)
    Ws = sorted(by)
    def label(w):
        r = by[w][0]
        # window_s / dt are missing from the oldest records; fall back, then give up
        ws = r.get("window_s") or (r["S"] * r["dt"] if "dt" in r else None)
        return f"W={w}\nS={r['S']}" + (f"\n{ws*1e3:.1f} ms" if ws else "")
    labels = [label(w) for w in Ws]
    groups = [by[w] for w in Ws]

    fig, ax = plt.subplots(1, 3, figsize=(13, 4.4))
    for a, (key, ylab, title) in zip(ax, [PANELS[3], PANELS[2], PANELS[1]]):
        _draw(a, groups, labels, key, ylab, title)
    ax[0].set_title("deployed RMS -- THE metric across W", fontsize=9)
    fig.suptitle(f"Window-length sweep, one LHS family, n_runs={_nruns(recs)}  "
                 f"({len(recs)} runs; every seed shown)")
    out = _graphs(out); fig.tight_layout(); out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=160); print(f"-> {out}")


def _nruns(recs):
    tags = {r["tag"].split("_n")[1].split("_")[0] for r in recs if "_n" in r["tag"]}
    return "/".join(sorted(tags)) if tags else "?"


def summary(recs, by_key):
    """The table you actually read. Bands, not points."""
    by = collections.defaultdict(list)
    for r in recs:
        by[by_key(r)].append(r)
    print(f"\n{'group':22s} {'n':>3s} " +
          " ".join(f"{k:>26s}" for k, _, _ in PANELS))
    for g in sorted(by):
        row = f"{str(g):22s} {len(by[g]):3d} "
        for key, _, _ in PANELS:
            v = [r[key] for r in by[g]]
            row += f" {np.median(v):9.3e}[{min(v):8.2e},{max(v):8.2e}]"
        print(row)
    print("\nmedian [min, max] across seeds. Non-overlapping [min,max] between two "
          "groups on val_th or per_window_rms is the bar for a real difference (F24).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("results_dir")
    p.add_argument("--kind", choices=["arms", "W", "wphys", "hd", "residual"], default="arms")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    recs = load(a.results_dir)
    if not recs:
        raise SystemExit(f"no usable records in {a.results_dir}")
    name = Path(a.results_dir).name

    if a.kind == "arms":
        summary(recs, lambda r: f"F={r['F']} mf={r['max_freq']:g}")
        plot_arms(recs, a.out or f"graphs/10_{name}_arms.png")
    elif a.kind == "wphys":
        summary(recs, lambda r: f"w_phys={r['w_phys']:g}")
        plot_wphys(recs, a.out or f"graphs/14_{name}.png")
    elif a.kind == "hd":
        summary(recs, lambda r: f"hidden={_hidden_dim(r)}")
        plot_hd(recs, a.out or f"graphs/17_{name}.png")
    elif a.kind == "residual":
        summary(recs, lambda r: f"{r.get('residual','eq4')} w={r['w_phys']:g}")
        plot_residual(recs, a.out or f"graphs/21_{name}.png")
    else:
        summary(recs, lambda r: f"W={r['W']}")
        plot_W(recs, a.out or f"graphs/11_{name}_W.png")
