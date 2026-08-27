#!/usr/bin/env python
"""Print the config lines that have NO result record yet, so a resubmit redoes only
the jobs that were lost instead of all of them.

    python hpc/pending.py hpc/exp1_w40_fourier.txt > hpc/exp1_missing.txt
    sh hpc/submit.sh hpc/exp1_missing.txt ffw40b

A record is written only on success (`train_pll.main` writes the JSON at the very end,
atomically), so "record exists" is exactly "this config is finished". Diverged runs
write a record with status != ok; those are counted as done here -- rerunning them
unchanged would just diverge again.

The tag is rebuilt with the same formula train_pll.main uses:
    {dataset stem}_n{n_runs}_W{W}_F{F}_mf{max_freq:g}_wp{w_phys:g}_s{seed}sp{split_seed}
n_runs and W come from the dataset's own meta, F and max_freq fall back to the YAML
defaults when the config line does not pass them.
"""
import argparse
import json
import shlex
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from paths import data as _data, sweeps as _sweeps

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_MODEL = OmegaConf.load(CONFIG_DIR / "DeepONet_models.yml")

_meta_cache = {}


def dataset_meta(path):
    if path not in _meta_cache:
        f = _data(path)
        if not f.exists():
            # n_runs and W come from the dataset itself, so the tag cannot be rebuilt
            # without it. The npz files live on the cluster (rsync excludes *.npz).
            sys.exit(f"dataset not found: {f}\nRun hpc/pending.py on the cluster, "
                     f"where the .npz files are.")
        z = np.load(f, allow_pickle=False)                    # npz is lazy: meta only
        _meta_cache[path] = json.loads(z["meta_json"].item())
    return _meta_cache[path]


def tag_for(line):
    a = shlex.split(line)
    get = lambda flag, default=None: (a[a.index(flag) + 1] if flag in a else default)
    ds = get("--dataset", "pll_dataset.npz")
    meta = dataset_meta(ds)
    F = int(get("--F", _MODEL.num_fourier_feats))
    mf = float(get("--max_freq", _MODEL.max_fourier_feat_frequency))
    wp = float(get("--w_phys", 0.0))
    seed = int(get("--seed", 0))
    sp = int(get("--split_seed", 0))
    # The suffixes train_pll appends, in the SAME order. Without them every gains job
    # (_g, since exp14) and every capacity job (_L, _w, since exp17) is reported as
    # missing and a resubmit reruns work that is already done.
    hd, nl, wd = get("--hidden_dim"), get("--n_layers"), get("--width")
    arch, res = get("--arch", "deeponet"), get("--residual", "eq4")
    tag = (f"{Path(ds).stem}_n{meta['n_runs']}_W{meta['W']}_F{F}"
           f"_mf{mf:g}_wp{wp:g}_s{seed}sp{sp}"
           + (f"_h{hd}" if hd else "")
           + ("" if arch == "deeponet" else f"_{arch}")
           + ("" if res == "eq4" else f"_{res}")
           + ("_g" if meta.get("gains", {}).get("enabled") else "")
           + (f"_L{nl}" if nl else "")
           + (f"_w{wd}" if wd else ""))
    return tag, get("--results_dir", "sweeps")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("configs")
    p.add_argument("--verbose", action="store_true", help="report to stderr")
    a = p.parse_args()

    lines = [l for l in Path(a.configs).read_text().splitlines()
             if l.strip() and not l.lstrip().startswith("#")]
    missing, done = [], 0
    for line in lines:
        tag, rdir = tag_for(line)
        if (_sweeps(rdir) / f"{tag}.json").exists():
            done += 1
        else:
            missing.append(line)

    for line in missing:
        print(line)
    if a.verbose or not missing:
        print(f"{done} done, {len(missing)} missing, of {len(lines)}", file=sys.stderr)
