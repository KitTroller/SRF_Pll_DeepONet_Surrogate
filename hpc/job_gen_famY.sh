#!/bin/sh
### famY -- the last corner of the noise factorial. ONE job, ~1 GB.
###
###     bsub < hpc/job_gen_famY.sh
###
### RESUBMITTING IS SAFE: skips the stem if the file already OPENS, regenerates it if it
### is a truncated zip.
###
### famY = limiter + FIXED gains + NO white noise, at --lhs_seed 21.
###
### n_runs IS 5000 AND MUST STAY 5000. The whole point of famY is that it pairs with
### famN, which is n=5000 at the same seed; scipy's LatinHypercube produces a different
### design for a different sample count, so a famY at n=10000 would pair with nothing
### and answer nothing. If the more-data question is worth running it is a separate
### family, not this one.
###
### WHAT IT COMPLETES. Seed 21 already holds three of the four (limiter x noise) corners
### at FIXED gains, and seed 22 holds all four at TUNABLE gains:
###
###                      seed 21, FIXED gains        seed 22, TUNABLE gains
###                      limiter    no limiter       limiter    no limiter
###     noise            famN       famR             famO       famX
###     no noise         famY <-    famV             famU       famW
###
### With famY the factorial is complete and the F68/exp24 result becomes attributable.
### As it stands the noise effect is +1.59x with fixed gains, 1.02x with tunable gains
### and a limiter, and 1.60x WORSE with tunable gains and no limiter -- three corners,
### two moving factors, and no clean way to separate them.
#BSUB -q hpc
#BSUB -J pll_gen_famY
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -M 3GB
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

f="data/famY_W40.npz"
case "$(npz_state "$f")" in
    ok)      echo "skip famY: present and readable"; exit 0 ;;
    corrupt) echo "removing truncated $f"; rm -f "$f" ;;
    missing) ;;
    *)       echo "cannot read $f and it is not a truncated zip -- stopping"; exit 1 ;;
esac

python hpc/generate_family.py --stem famY --W 40 --n_runs 5000 --lhs_seed 21 \
       --freq_limit 18.8496 --no_white_noise || exit 1

echo "done:"
ls -la data/famY_W40.npz
getquota_zhome.sh 2>/dev/null

# The pairing claim, checked rather than asserted. famY must differ from famN ONLY by the
# noise: same ICs, same grid waveform up to the noise term, both limited, gains off.
python - <<'PY'
import json
import numpy as np
y, n = np.load("data/famY_W40.npz"), np.load("data/famN_W40.npz")
for nm, z in (("famY", y), ("famN", n)):
    m = json.loads(z["meta_json"].item())       # meta_json, NOT meta: savez stores arrays
    print(f"{nm}  white_noise={m.get('white_noise', '(pre-flag)')}  "
          f"freq_limit={m.get('freq_limit')}  gains_stored={'kp' in z.files}")
for k in ("lhs_samples", "fault_kind"):
    if k in y.files and k in n.files:
        print(f"  {k:12s} bit-equal: {np.array_equal(y[k], n[k])}")
print(f"  Va max |difference|: {np.abs(y['Va'] - n['Va']).max():.5f}  (0.05000 = the noise)")
PY
