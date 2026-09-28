"""What does making Kp/Ki INPUTS cost, at the operating point, like-for-like, on the LIMITED system?

    python src/analysis/gains_cost.py            # ~2 min

F57's 3.5x is UNLIMITED physics at L2_w64, averaged over the gain box. graphs/31 (F78) gives
1.42x at 25/300, but in the warm regime only (omega0 +/-2) and one seed per model. This is
the paper's number: famN (limited, FIXED 25/300) against famO (limited, TUNABLE, fed 25/300),
both L3_w128, n=5000, no anchor, every seed on disk, scored on the SAME limited truth at
Kp=25, Ki=300 over the full training envelope (omega0 +/-20), clean + sag + jump runs.
The flagship is scored on the same runs for reference (NOT like-for-like: 40k + anchor).
"""

# src/ on the path: these scripts live in src/analysis/ but import the pipeline
# modules (paths, PLL_Simulator, train_pll, sweep) that stay in src/. Running
# `python src/analysis/foo.py` puts src/analysis on sys.path, not src/.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import argparse
import glob
import statistics as st

import PLL_Simulator as PS
from common_test import load_f32, truth, rollout_rms
from paths import ROOT

LIMIT = 18.8496
GROUPS = {
    "FIXED 25/300 (famN)": "runs/famN_W40_n5000_W40_F4_mf503_wp0.3_s*sp0_L3_w128.pth",
    "TUNABLE (famO)":      "runs/famO_W40_n5000_W40_F4_mf503_wp0.3_s[0-7]sp0_L3_w128_g.pth",
    "flagship (famO40k s6)": "runs/famO40k_W40_n40000_W40_F4_mf503_wp0.3_s6sp0_L3_w128_ao_g.pth",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_runs", type=int, default=32, help="per kind (clean, sag, jump)")
    a = p.parse_args()
    PS.pll_constants.freq_limit = LIMIT
    kp, ki = 25.0, 300.0
    cases = {k: truth(kp, ki, a.n_runs, kind=k, omega0=20.0) for k in ("clean", "sag", "jump")}

    res = {}
    for g, pat in GROUPS.items():
        paths = sorted(glob.glob(str(ROOT / pat)))
        res[g] = {k: [] for k in cases}
        for c in paths:
            model, ck = load_f32(c)
            for k, (Va, Vb, Vc, th, om) in cases.items():
                res[g][k].append(rollout_rms(model, ck, (Va, Vb, Vc), th, om, kp, ki))
        print(f"{g:24s} {len(paths)} seeds: " + "  ".join(
            f"{k} {st.median(v):.3e} [{min(v):.2e}-{max(v):.2e}]" for k, v in res[g].items()), flush=True)

    fx, tu = res["FIXED 25/300 (famN)"], res["TUNABLE (famO)"]
    print("\ncost of tunability at 25/300 (median tunable / median fixed, >1 = tunable is worse):")
    for k in cases:
        print(f"  {k:6s} {st.median(tu[k]) / st.median(fx[k]):.2f}x   "
              f"(flagship / fixed: {st.median(res['flagship (famO40k s6)'][k]) / st.median(fx[k]):.2f}x)")


if __name__ == "__main__":
    main()
