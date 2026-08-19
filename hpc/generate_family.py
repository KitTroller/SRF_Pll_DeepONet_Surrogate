#!/usr/bin/env python
"""Generate ONE LHS family at several windowings, with a clobber guard.

Why the guard: `create_initial_condition_space` calls scipy's LatinHypercube with
no seed, so a dataset that is overwritten CANNOT be reproduced. Every sweep record
naming that file becomes un-revaluable, and every checkpoint trained on it becomes
un-scoreable. `Dataset_Creator.generate_multi_W`'s default path_fmt is
"pll_dataset_W{W}.npz", which is exactly the n=1000 family that sweeps_Wtest lives on.

    python hpc/generate_family.py --stem famB --W 10 20 40 100

writes famB_W10.npz ... famB_W100.npz using n_runs from config/initial_conditions.yml.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paths import data as _data
from dataset_generator import Dataset_Creator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stem", required=True, help="famB -> famB_W40.npz")
    p.add_argument("--W", type=int, nargs="+", default=[10, 20, 40, 100])
    p.add_argument("--outdir", default=None,
                   help="default: data/ , the same folder save_dataset writes to")
    p.add_argument("--force", action="store_true",
                   help="overwrite existing files (they cannot be regenerated)")
    a = p.parse_args()

    # The guard MUST test the path save_dataset actually writes to. After the src/data
    # reorg, `save_dataset` resolves a bare name through paths.data() into data/, so a
    # guard that checked the project root would never fire and the whole point of it
    # -- protecting unreproducible datasets -- would be silently lost.
    fmt = str(_data(a.stem + "_W{W}.npz") if a.outdir is None
              else Path(a.outdir) / (a.stem + "_W{W}.npz"))
    clash = [fmt.format(W=W) for W in a.W if Path(fmt.format(W=W)).exists()]
    if clash and not a.force:
        sys.exit("refusing to overwrite:\n  " + "\n  ".join(clash) +
                 "\n\nThe LHS draw is not seeded, so these files are unreproducible and "
                 "every sweep record that names them\nwould become un-revaluable. "
                 "Pick a different --stem, or pass --force if you really mean it.")

    dc = Dataset_Creator()
    sim = dc.pll_simulator
    print(f"n_runs={dc.n_runs}  N={sim.N}  dt={sim.dt}  "
          f"time_window={sim.physics.time_window}s")
    for W in a.W:
        assert sim.N % W == 0, f"W={W} does not divide N={sim.N}"
    print(f"writing: {', '.join(fmt.format(W=W) for W in a.W)}")
    dc.generate_multi_W(a.W, path_fmt=fmt)


if __name__ == "__main__":
    main()
