#!/bin/sh
### exp34 data: famO40k at W=20 (25 ms) AND W=50 (10 ms), one solve, two files, ~5.2 GB each, --slim.
###
###     bsub < hpc/job_gen_exp34.sh
###
### WHY REGENERATE and not src/rewindow.py: rewindow predates tunable gains. It reshapes every
### array through (n_runs, N), but kp/ki are stored per WINDOW, shape (n_runs*W,), so it would
### crash on this family -- and W=50 is not reachable from W=40 by merging or splitting anyway
### (S=125 -> 100). generate_multi_W does ONE seeded LHS draw and ONE solve and only then
### slices, so at the same --lhs_seed 28 the runs should come out identical to famO40k_W40's.
### Verified on a 40-run miniature (2026-09-29); the CHECK lines at the end verify it again on
### the real files, run by run.
###
### DISK (30 GB home, 26.45 used on 2026-09-29). Both files fit only after the seven deletions in
### hpc/exp34a_resweep.txt: three seeded families (regenerable) and the four famH-K_W20 families
### (irreplaceable -- backed up to the laptop and checksum-verified FIRST). The guard below tests
### for those files rather than the quota, because getquota_zhome.sh is refreshed only every 4 h.
###
### RESUBMITTING IS SAFE: a file that already OPENS is kept, a truncated zip is regenerated.
### -W 2:00 and 6 GB/core x 8: as hpc/job_gen_famO40k.sh (~25 GB at the fine step before
### windowing), with slack for the second save.
#BSUB -J pll_gen_exp34
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=6GB]"
#BSUB -M 7GB
#BSUB -W 2:00
#BSUB -o logs/gen_%J.out
#BSUB -e logs/gen_%J.err

cd "$LS_SUBCWD" || exit 1
. .venv/bin/activate

export OMP_NUM_THREADS=${LSB_DJOB_NUMPROC:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONUNBUFFERED=1

for b in famO20k_W40 famQ_W40 famO_W40 famH_W20 famI_W20 famJ_W20 famK_W20; do
    [ -e "data/$b.npz" ] && { echo "refusing: data/$b.npz still exists, so both new families will not fit -- see hpc/exp34a_resweep.txt"; exit 1; }
done
[ -e data/famO40k_W40.npz ] || { echo "data/famO40k_W40.npz missing -- the CHECK needs it"; exit 1; }

echo "host $(hostname)  cores $OMP_NUM_THREADS  cwd $(pwd)"
getquota_zhome.sh 2>/dev/null || echo "getquota_zhome.sh unavailable"

npz_state () {                                  # -> ok | corrupt | missing | unknown
    python - "$1" <<'PY'
import os, sys, zipfile
import numpy as np
p = sys.argv[1]
if not os.path.exists(p):
    print("missing")
else:
    try:
        np.load(p)
        print("ok")
    except zipfile.BadZipFile:
        print("corrupt")
    except Exception:
        print("unknown")
PY
}

TODO=""
for W in 20 50; do
    f="data/famO40k_W$W.npz"
    case "$(npz_state "$f")" in
        ok)      echo "keep: $f present and readable" ;;
        corrupt) echo "removing truncated $f"; rm -f "$f"; TODO="$TODO $W" ;;
        missing) TODO="$TODO $W" ;;
        *)       echo "cannot read $f and it is not a truncated zip -- stopping"; exit 1 ;;
    esac
done

# identical flags to hpc/job_gen_famO40k.sh; only --W differs
# shellcheck disable=SC2086
[ -z "$TODO" ] || python hpc/generate_family.py --stem famO40k --W $TODO --n_runs 40000 --lhs_seed 28 \
                         --freq_limit 18.8496 --gains --slim || exit 1

echo "done:"
ls -la data/famO40k_W20.npz data/famO40k_W50.npz
getquota_zhome.sh 2>/dev/null

# "the same 40000 runs, sliced differently", checked on the real files rather than asserted.
for W in 20 50; do
python - "$W" <<'PY'
import json, sys
import numpy as np
W = int(sys.argv[1])
n, o = np.load(f"data/famO40k_W{W}.npz"), np.load("data/famO40k_W40.npz")
mn, mo = json.loads(n["meta_json"].item()), json.loads(o["meta_json"].item())
print(f"CHECK W={W} meta keys that differ from famO40k_W40:",
      sorted(k for k in set(mn) | set(mo) if mn.get(k) != mo.get(k)), " (want: S, W)")
R = mn["n_runs"]
for k in ("theta_pll", "omega_pll", "Va", "Vb", "Vc"):   # one array at a time: 1.6 GB each in float64
    a, b = n[k].reshape(R, -1), o[k].reshape(R, -1)
    print(f"CHECK W={W} {k:9s} run by run: identical {np.array_equal(a, b)}  max|diff| {np.abs(a - b).max():.1e}")
    del a, b
for k in ("lhs_samples", "disturbance"):
    print(f"CHECK W={W} {k:11s} identical {np.array_equal(n[k], o[k])}")
for k in ("kp", "ki"):                                   # stored per window: compare one per run
    print(f"CHECK W={W} {k} per run identical {np.array_equal(n[k][::W], o[k][::40])}")
print(f"CHECK W={W} slim", mn.get("slim"), "| freq_limit", mn["freq_limit"], "| lhs_seed", mn["lhs_seed"],
      "| W", mn["W"], "S", mn["S"])
PY
done
