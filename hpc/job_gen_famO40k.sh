#!/bin/sh
### famO40k -- famO's physics at n_runs 40000 (8x famO, 2x famO20k). ONE job, ~5.2 GB. Feeds exp33.
###
###     bsub < hpc/job_gen_famO40k.sh
###
### NEEDS ~6 GB FREE in home, and --slim is what makes it fit (a full save would be 8.1 GB).
### Free space first: see docs/notes.md, "DATASETS DELETED".
###
### RESUBMITTING IS SAFE: skips the stem if the file already OPENS, regenerates it if it
### is a truncated zip.
###
### Same flags as famO/famQ/famO20k (limiter, tunable gains, noise, faults), --lhs_seed 28,
### a fresh seed. A different n is a different LHS design anyway, so no pairing is implied.
###
### --slim drops Vd/Vq/Valpha/Vbeta; load_dataset rebuilds them exactly from Va/Vb/Vc and
### theta_pll (Park/Clarke). Verified 2026-09-23 on a 40-run pair: every other array
### bit-identical, the four rebuilt ones agree to 1.2e-7 = float32 resolution, 35% smaller.
### Cost: a few seconds and ~11 GB of float64 scratch per load -- hence the 8 GB/core the
### TRAINING jobs ask for in hpc/job_sweep_gpu_long.sh.
###
### -W 1:30 and 6 GB/core x 8: generation holds every run at the fine step before windowing,
### ~25 GB at n=40000 (12 GB at 20000, where 4 GB/core sufficed). ~14 min by the famS+famT
### rate (2 x 5000 runs in 198 s).
#BSUB -J pll_gen_famO40k
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=6GB]"
#BSUB -M 7GB
#BSUB -W 1:30
#BSUB -o logs/gen_%J.out
#BSUB -e logs/gen_%J.err

cd "$LS_SUBCWD" || exit 1
. .venv/bin/activate

export OMP_NUM_THREADS=${LSB_DJOB_NUMPROC:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONUNBUFFERED=1

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

f="data/famO40k_W40.npz"
case "$(npz_state "$f")" in
    ok)      echo "skip famO40k: present and readable"; exit 0 ;;
    corrupt) echo "removing truncated $f"; rm -f "$f" ;;
    missing) ;;
    *)       echo "cannot read $f and it is not a truncated zip -- stopping"; exit 1 ;;
esac

python hpc/generate_family.py --stem famO40k --W 40 --n_runs 40000 --lhs_seed 28 \
       --freq_limit 18.8496 --gains --slim || exit 1

echo "done:"
ls -la data/famO40k_W40.npz
getquota_zhome.sh 2>/dev/null

# "famO's physics at 4x the runs", checked on the real files rather than asserted.
python - <<'PY'
import json
import numpy as np
n, o = np.load("data/famO40k_W40.npz"), np.load("data/famO_W40.npz")
mn, mo = json.loads(n["meta_json"].item()), json.loads(o["meta_json"].item())
print("CHECK meta keys that differ from famO:", sorted(k for k in set(mn) | set(mo) if mn.get(k) != mo.get(k)),
      " (want: lhs_seed, n_runs, slim, white_noise)")
print("CHECK n_runs", mn["n_runs"], "| freq_limit", mn["freq_limit"], "| gains", "kp" in n.files,
      "| slim", mn.get("slim"), "| Vq dropped:", "Vq" not in n.files)
PY
