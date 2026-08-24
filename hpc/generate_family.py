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
    p.add_argument("--n_runs", type=int, default=None,
                   help="override config/initial_conditions.yml")
    p.add_argument("--sensors", type=int, default=None,
                   help="override config/PLL_Constants.yml. sensors/time_window IS dt, "
                        "so --sensors 10000 over a 0.5 s window means dt = 50 us")
    p.add_argument("--lhs_seed", type=int, default=None,
                   help="make the dataset REPRODUCIBLE. Without it the Latin-Hypercube draw "
                        "and the sensor noise are unseeded and the file can never be "
                        "regenerated -- which is why the clobber guard exists. Always pass it.")
    p.add_argument("--gains", action="store_true",
                   help="sample Kp and Ki per run and store them, so the network takes "
                        "them as INPUTS. Ranges come from config/initial_conditions.yml")
    p.add_argument("--kp_range", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                   help="override gains.Kp. Implies --gains. Shrinking the box trades "
                        "controller coverage for accuracy inside what is left")
    p.add_argument("--ki_range", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                   help="override gains.Ki. Implies --gains")
    p.add_argument("--no_faults", action="store_true",
                   help="disable sags and phase jumps, so all n_runs are clean. The run "
                        "budget is unchanged, so the clean regime is sampled twice as "
                        "densely (config fraction is 0.5). The model then has NEVER seen "
                        "a fault waveform -- fine only if the faults come from the "
                        "co-simulation's own network solver, not from us")
    p.add_argument("--omega_range", type=float, default=None,
                   help="half-range for the PLL's initial omega, default 20 rad/s. "
                        "the EMT co-simulation never leaves |omega| < 0.15, and only 6.8%% "
                        "of our windows are in that band -- a narrow model is a specialist "
                        "for warm co-simulation, not a replacement for the wide one")
    a = p.parse_args()

    # Override in memory rather than editing the YAMLs. Editing them would change the
    # defaults for every other process on this machine -- including anything already
    # running -- and is the kind of thing that gets forgotten and silently poisons the
    # next family. NOTE both modules call OmegaConf.load separately, so they hold
    # DIFFERENT objects and both have to be patched.
    gains = a.gains or a.kp_range is not None or a.ki_range is not None
    if (a.n_runs is not None or a.sensors is not None or a.omega_range is not None
            or gains or a.no_faults):
        import dataset_generator as DG
        import PLL_Simulator as PS
        if a.n_runs is not None:
            DG.initial_conditions_config.n_runs = a.n_runs
            PS.initial_conditions_config.n_runs = a.n_runs
        if a.sensors is not None:
            PS.pll_constants.sensors = a.sensors
        if a.omega_range is not None:
            # only Dataset_Creator reads `ranges`; PLL_Simulator never does.
            DG.initial_conditions_config.ranges.omega_pll = [-a.omega_range, a.omega_range]
        if gains:
            DG.initial_conditions_config.gains.enabled = True
            if a.kp_range is not None:
                DG.initial_conditions_config.gains.Kp = list(a.kp_range)
            if a.ki_range is not None:
                DG.initial_conditions_config.gains.Ki = list(a.ki_range)
        if a.no_faults:
            # `create_disturbance_space` returns before it touches the RNG when this is
            # off, so two no-fault families sharing an --lhs_seed get IDENTICAL initial
            # conditions and IDENTICAL gain u-draws -- only the affine map onto (Kp,Ki)
            # differs. That makes a gain-box comparison PAIRED instead of two independent
            # draws, which is most of the reason this batch can use 4 seeds and not 16.
            DG.initial_conditions_config.disturbances.enabled = False

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

    dc = Dataset_Creator(seed=a.lhs_seed)
    sim = dc.pll_simulator
    print(f"n_runs={dc.n_runs}  N={sim.N}  dt={sim.dt}  "
          f"time_window={sim.physics.time_window}s  lhs_seed={a.lhs_seed}")
    g, d = dc.init_cond.gains, dc.init_cond.disturbances
    print(f"  faults={'ON' if d.enabled else 'OFF'}   "
          f"gains={'Kp %s  Ki %s' % (list(g.Kp), list(g.Ki)) if g.enabled else 'fixed'}")
    if a.lhs_seed is None:
        print("  !! no --lhs_seed: this dataset will NOT be reproducible")
    for W in a.W:
        assert sim.N % W == 0, f"W={W} does not divide N={sim.N}"
    print(f"writing: {', '.join(fmt.format(W=W) for W in a.W)}")
    dc.generate_multi_W(a.W, path_fmt=fmt)


if __name__ == "__main__":
    main()
