#!/bin/sh
### famZ -- famO with TWICE THE DATA. ONE job, ~2 GB. Feeds exp29.
###
###     bsub < hpc/job_gen_famZ.sh
###
### RESUBMITTING IS SAFE: skips the stem if the file already OPENS, regenerates it if it
### is a truncated zip.
###
### famZ = limiter + TUNABLE gains + noise + faults, exactly famO's physics, at
### n_runs 10000 instead of 5000. Verified before writing this (20-run local test,
### 2026-09-18): its meta differs from famO_W40.npz in lhs_seed, n_runs and nothing else
### (famO predates the white_noise key; noise was on then too).
###
### --lhs_seed 26, A FRESH SEED, deliberately NOT famO's 22. scipy's LatinHypercube draws
### a different design for a different sample count, so famZ cannot be paired with famO
### whatever the seed -- reusing 22 would only suggest a pairing that does not exist.
### Seeds 11, 21-25 are taken.
###
### SIZE: 20 runs wrote 4.0 MB, so 10000 is ~2.0 GB. Check the quota line this prints.
### MEMORY: the generator holds every run at the fine step before windowing, so the peak
### is well above the file size -- 3 GB/core x 8 here, up from famY's 2 GB at n=5000.
#BSUB -q hpc
#BSUB -J pll_gen_famZ
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=3GB]"
#BSUB -M 4GB
#BSUB -W 24:00
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

f="data/famZ_W40.npz"
case "$(npz_state "$f")" in
    ok)      echo "skip famZ: present and readable"; exit 0 ;;
    corrupt) echo "removing truncated $f"; rm -f "$f" ;;
    missing) ;;
    *)       echo "cannot read $f and it is not a truncated zip -- stopping"; exit 1 ;;
esac

python hpc/generate_family.py --stem famZ --W 40 --n_runs 10000 --lhs_seed 26 \
       --freq_limit 18.8496 --gains || exit 1

echo "done:"
ls -la data/famZ_W40.npz
getquota_zhome.sh 2>/dev/null

# The "same physics as famO" claim, checked on the real file rather than asserted.
python - <<'PY'
import json
import numpy as np
z, o = np.load("data/famZ_W40.npz"), np.load("data/famO_W40.npz")
mz, mo = json.loads(z["meta_json"].item()), json.loads(o["meta_json"].item())
diff = sorted(k for k in set(mz) | set(mo) if mz.get(k) != mo.get(k))
print("meta keys that differ from famO:", diff, " (want: lhs_seed, n_runs, white_noise)")
print("n_runs", mz["n_runs"], " freq_limit", mz["freq_limit"], " gains stored", "kp" in z.files)
PY
