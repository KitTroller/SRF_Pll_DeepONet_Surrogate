#!/bin/sh
### famO20k -- famO's physics at n_runs 20000 (4x famO, 2x famQ). ONE job, ~4 GB. Feeds exp32.
###
###     bsub < hpc/job_gen_famO20k.sh
###
### NEEDS ~4.5 GB FREE in home. Free it first (see docs/notes.md, "Datasets deleted
### 2026-09-21"): the quota is 30 GB and data/ alone was ~25 GB.
###
### RESUBMITTING IS SAFE: skips the stem if the file already OPENS, regenerates it if it
### is a truncated zip.
###
### Same flags as famO/famQ (limiter, tunable gains, noise, faults), --lhs_seed 27 -- a FRESH
### seed: 11, 21-25 are taken, 26 was named for a family that was never generated. A
### different n is a different LHS design anyway, so no pairing with famO/famQ is implied.
###
### -W 0:45: famS+famT (2 x 5000 runs) took 198 s, so 20000 runs is ~7 min. 4 GB/core x 8:
### the generator holds every run at the fine step before windowing; famY needed 2 GB/core
### at n=5000.
#BSUB -q hpc
#BSUB -J pll_gen_famO20k
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 5GB
#BSUB -W 0:45
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

f="data/famO20k_W40.npz"
case "$(npz_state "$f")" in
    ok)      echo "skip famO20k: present and readable"; exit 0 ;;
    corrupt) echo "removing truncated $f"; rm -f "$f" ;;
    missing) ;;
    *)       echo "cannot read $f and it is not a truncated zip -- stopping"; exit 1 ;;
esac

python hpc/generate_family.py --stem famO20k --W 40 --n_runs 20000 --lhs_seed 27 \
       --freq_limit 18.8496 --gains || exit 1

echo "done:"
ls -la data/famO20k_W40.npz
getquota_zhome.sh 2>/dev/null

# "famO's physics at 4x the runs", checked on the real files rather than asserted.
python - <<'PY'
import json
import numpy as np
n, o = np.load("data/famO20k_W40.npz"), np.load("data/famO_W40.npz")
mn, mo = json.loads(n["meta_json"].item()), json.loads(o["meta_json"].item())
print("CHECK meta keys that differ from famO:", sorted(k for k in set(mn) | set(mo) if mn.get(k) != mo.get(k)),
      " (want: lhs_seed, n_runs, white_noise)")
print("CHECK n_runs", mn["n_runs"], "| freq_limit", mn["freq_limit"], "| gains stored", "kp" in n.files)
PY
