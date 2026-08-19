"""Derive a new windowing from an existing dataset, WITHOUT regenerating.

    python src/rewindow.py famB_W10.npz --W 50

`generate_multi_W` fixes the family by doing one LHS draw and one ODE solve, then
slicing it several ways. But the W list is baked in at generation time, and the raw
solve is not saved -- so adding a W later would need a fresh generate, which draws a
NEW (unseeded) LHS and destroys the family.

It does not have to. A window is a contiguous slice of its run, so a file at W_old
(S_old samples per window) can be re-sliced to any W_new whose S_new divides S_old:
just split each stored window into S_old/S_new consecutive pieces. The samples, the
noise realisation, the ICs and the solve are all bit-identical -- only the bookkeeping
changes. Verified by deriving W=20 from W=10 and diffing against the real W=20 file.

Constraint: S_new must divide S_old, i.e. W_new must be a multiple of W_old.
From W=10 (S=500): W = 20, 25, 50, 100, 125, 250, 500.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from paths import data as _data

# (n_samples, S) arrays that get re-sliced; everything else is rebuilt or copied
_PER_WINDOW_SKIP = {"t_local", "theta0", "omega0", "run_id", "segment_id",
                    "lhs_samples", "disturbance", "fault_kind", "window_faulted"}


def rewindow(src, W_new, out=None):
    z = np.load(_data(src), allow_pickle=False)
    meta = json.loads(z["meta_json"].item())
    W_old, S_old, n_runs = meta["W"], meta["S"], meta["n_runs"]
    S_new = meta["N"] // W_new
    if S_old % S_new:
        raise SystemExit(f"S_new={S_new} does not divide S_old={S_old}: "
                         f"W_new={W_new} must be a multiple of W_old={W_old}")
    k = S_old // S_new                                   # pieces per stored window

    out_arrays = {}
    for name in z.files:
        if name == "meta_json" or name in _PER_WINDOW_SKIP:
            continue
        a = z[name]                                       # (n_runs*W_old, S_old)
        # (runs, W_old, S_old) -> (runs, W_old, k, S_new) -> (runs*W_new, S_new).
        # C-order keeps time in order: window w piece p becomes new window w*k + p.
        out_arrays[name] = a.reshape(n_runs, W_old, k, S_new).reshape(-1, S_new)

    dt = meta["dt"]
    out_arrays["t_local"]    = (np.arange(S_new) * dt).astype(np.float64)
    out_arrays["theta0"]     = out_arrays["theta_pll"][:, 0].astype(np.float64)
    out_arrays["omega0"]     = out_arrays["omega_pll"][:, 0].astype(np.float64)
    out_arrays["run_id"]     = np.repeat(np.arange(n_runs), W_new).astype(np.int32)
    out_arrays["segment_id"] = np.tile(np.arange(W_new), n_runs).astype(np.int32)
    out_arrays["lhs_samples"] = z["lhs_samples"]

    if "disturbance" in z.files:                          # per-run: carried over as is
        d = z["disturbance"]
        out_arrays["disturbance"] = d
        out_arrays["fault_kind"] = np.repeat(z["fault_kind"][::W_old], W_new).astype(np.int32)
        # the per-window flag has to be recomputed on the NEW window grid
        w_start = np.arange(W_new) * S_new * dt
        w_end = w_start + S_new * dt
        kind = z["fault_kind"][::W_old][:, None]
        t0s, durs, t0j = d[:, 0:1], d[:, 1:2], d[:, 3:4]
        sag_hit = (kind == 1) & (t0s < w_end) & (t0s + durs > w_start)
        jump_hit = (kind == 2) & (t0j >= w_start) & (t0j < w_end)
        out_arrays["window_faulted"] = (sag_hit | jump_hit).reshape(-1).astype(np.int32)

    meta["W"], meta["S"] = W_new, S_new
    meta["derived_from"] = str(src)                       # provenance: same family
    out_arrays["meta_json"] = np.array(json.dumps(meta))

    out = out or f"{Path(src).stem.rsplit('_W', 1)[0]}_W{W_new}.npz"
    dst = _data(out)
    if dst.exists():
        raise SystemExit(f"refusing to overwrite {dst}")
    np.savez_compressed(dst, **out_arrays)
    print(f"saved {dst}  n_runs={n_runs} W={W_new} S={S_new}  "
          f"({out_arrays['theta_pll'].shape[0]} samples x {S_new})")
    return dst


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("--W", type=int, required=True, dest="W_new")   # must match the signature
    p.add_argument("--out", default=None)
    rewindow(**vars(p.parse_args()))
