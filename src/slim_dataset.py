"""Shrink a dataset by dropping what can be recomputed exactly.

    python src/slim_dataset.py famD_W40.npz            # -> famD_W40_slim.npz

Four of the nine bulk arrays are exact functions of the others:

    Vd, Vq       = park_dqTransform(Va, Vb, Vc, theta_pll)   elementwise, same index
    Valpha,Vbeta = clarke(Va, Vb, Vc)                        no angle involved

`simulate_batch` stores `Vd[:,k], Vq[:,k]` computed from `Va[:,k], Vb[:,k], Vc[:,k]` and
`theta[:,k]`, so the reconstruction is the same expression on the same inputs -- exact in
exact arithmetic. Measured against the originals it agrees to **1e-7 relative**, which is
float32 epsilon: the stored copies were float32, so the residue is the precision they were
saved at, not a modelling error. For scale, Vq enters the physics loss through
`s2 = RMS(Ki*Vq) = 65.6`, so 1e-7 of Vq is ~3e-8 absolute against a target RMS of ~1e-4.
Irrelevant, but say "1e-7 relative", not "bit-exact".

`omega_pll` additionally drops to float32: it is O(20) rad/s, so float32 resolves it to
~1e-6, and every consumer calls `.float()` on it anyway. **`theta_pll` stays float64** --
it is O(150) rad while the quantity that matters is the ~1e-4 deviation left after
subtracting the ramp (D5), and float32 would eat most of that in cancellation.

Together: ~44% smaller. `Dataset_Creator.load_dataset` rebuilds the missing arrays
transparently, so a slim file is a drop-in for a full one.
"""
import argparse
import json

import numpy as np

from paths import data as _data

DERIVABLE = ["Vd", "Vq", "Valpha", "Vbeta"]


def slim(src, out=None):
    z = np.load(_data(src), allow_pickle=False)
    meta = json.loads(z["meta_json"].item())
    keep = {}
    for k in z.files:
        if k in DERIVABLE:
            continue
        a = z[k]
        keep[k] = a.astype(np.float32) if k == "omega_pll" else a
    meta["slim"] = True                       # load_dataset keys off this
    keep["meta_json"] = np.array(json.dumps(meta))
    out = out or f"{str(src).replace('.npz','')}_slim.npz"
    dst = _data(out)
    if dst.exists():
        raise SystemExit(f"refusing to overwrite {dst}")
    np.savez_compressed(dst, **keep)
    import os
    a, b = os.path.getsize(_data(src)), os.path.getsize(dst)
    print(f"{src} {a/1e6:.0f} MB  ->  {out} {b/1e6:.0f} MB   ({100*(1-b/a):.0f}% smaller)")
    return dst


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("src"); p.add_argument("--out", default=None)
    a = p.parse_args()
    slim(a.src, a.out)
