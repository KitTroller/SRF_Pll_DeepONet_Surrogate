#!/bin/sh
### famO70 -- famO with phase jumps up to +/-70 deg instead of +/-60. ONE job, ~1 GB. Feeds exp31.
###
###     bsub < hpc/job_gen_famO70.sh
###
### RESUBMITTING IS SAFE: skips the stem if the file already OPENS, regenerates it if it
### is a truncated zip.
###
### BIT-PAIRED WITH famO: same --lhs_seed 22, same n_runs, same flags, only --jump_deg 70.
### The angle is lo + (hi-lo)*u with u from the seeded LHS, so every jump is famO's jump
### x 70/60 and NOTHING else moves -- ICs, gains, sags, fault assignment, noise all
### bit-identical. Verified on a 40-run pair (2026-09-18): meta differs only in
### `disturbances`; ratio exactly 1.166667; Va bit-equal on every run without a jump. The
### block at the end re-checks it on the real 5000-run files.
###
### -W 0:30, NOT 24:00. Every past generation took ~3 min (famS+famT 198 s, famW+famX 161 s,
### famY 174 s). The 24 h requests were never needed and block a job from starting before
### a service window.
#BSUB -q hpc
#BSUB -J pll_gen_famO70
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -M 3GB
#BSUB -W 0:30
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

f="data/famO70_W40.npz"
case "$(npz_state "$f")" in
    ok)      echo "skip famO70: present and readable"; exit 0 ;;
    corrupt) echo "removing truncated $f"; rm -f "$f" ;;
    missing) ;;
    *)       echo "cannot read $f and it is not a truncated zip -- stopping"; exit 1 ;;
esac

python hpc/generate_family.py --stem famO70 --W 40 --n_runs 5000 --lhs_seed 22 \
       --freq_limit 18.8496 --gains --jump_deg 70 || exit 1

echo "done:"
ls -la data/famO70_W40.npz
getquota_zhome.sh 2>/dev/null

# The pairing claim, checked on the real files rather than asserted.
python - <<'PY'
import json
import numpy as np
n, o = np.load("data/famO70_W40.npz"), np.load("data/famO_W40.npz")
mn, mo = json.loads(n["meta_json"].item()), json.loads(o["meta_json"].item())
print("meta keys that differ:", sorted(k for k in set(mn) | set(mo) if mn.get(k) != mo.get(k)),
      " (want: disturbances, white_noise -- famO predates the white_noise key)")
W = mn["W"]
J = o["fault_kind"][::W] == 2
r = n["disturbance"][J, 4] / o["disturbance"][J, 4]
print(f"jump runs {int(J.sum())}; angle ratio min {r.min():.6f} max {r.max():.6f}  (want 1.166667)")
for k in ("lhs_samples", "fault_kind", "kp", "ki"):
    print(f"  {k:12s} bit-equal: {np.array_equal(n[k], o[k])}")
nj = np.repeat(~J, W)
print("  Va bit-equal on runs WITHOUT a jump:", np.array_equal(n["Va"][nj], o["Va"][nj]))
PY
