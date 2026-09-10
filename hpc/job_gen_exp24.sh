#!/bin/sh
### exp24's two families. ONE job.
###
###     bsub < hpc/job_gen_exp24.sh
###
### RESUBMITTING IS SAFE: `gen` skips a stem whose files all OPEN, and deletes and
### regenerates any that are truncated. Existence is not readability -- a job killed
### mid-savez leaves a file that stats fine and raises BadZipFile on load.
###
### BOTH CARRY --lhs_seed 22, THE SAME SEED AS famO AND famU, and both carry --gains.
### That is the whole point: with gains held on and the seed held fixed, the four
### families form a closed 2x2 in (limiter, white noise) whose cells are bit-paired.
###
###                     limiter        no limiter
###     noise           famO           famX   <- new
###     no noise        famU           famW   <- new
###
### famJ_W40 is the family this would otherwise be compared against -- unlimited, gains,
### noise -- but it was generated with NO --lhs_seed, so it is unreproducible and cannot
### be paired with anything. famX exists to replace it inside this grid.
###
### DISK: ~2 GB for the pair, home is 30 GB. Check `getquota_zhome.sh` first.
#BSUB -q hpc
#BSUB -J pll_gen_e24
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
# NOT `df`: home is quota-limited, and df reports the shared filesystem.
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

gen () {                                        # gen <stem> <W list> <extra args...>
    stem=$1; Ws=$2; shift 2
    todo=""
    for W in $Ws; do
        f="data/${stem}_W${W}.npz"
        case "$(npz_state "$f")" in
            ok)      ;;
            missing) todo="$todo $W" ;;
            corrupt) echo "removing truncated $f"; rm -f "$f"; todo="$todo $W" ;;
            *)       echo "cannot read $f and it is not a truncated zip -- stopping"; return 1 ;;
        esac
    done
    if [ -z "$todo" ]; then echo "skip ${stem}: all windowings present and readable"; return 0; fi
    echo "generating ${stem} at W =${todo}"
    # shellcheck disable=SC2086
    python hpc/generate_family.py --stem "$stem" --W $todo "$@"
}

# NO --freq_limit on either line. Omitting it leaves pll_constants.freq_limit at its
# YAML default of None and simulate_batch takes the plain trapezoid branch, which is
# bit-identical to every unlimited family in the project (verified to 8.2e-13 rad).
gen famW "40" --n_runs 5000 --lhs_seed 22 --gains --no_white_noise || exit 1
gen famX "40" --n_runs 5000 --lhs_seed 22 --gains                  || exit 1

echo "done:"
ls -la data/fam[WX]_W40.npz
getquota_zhome.sh 2>/dev/null

# The pairing claim, checked rather than asserted. famW/famX must differ ONLY by the
# noise, and both must be unlimited with gains stored.
python - <<'PY'
import json
import numpy as np
w, x = np.load("data/famW_W40.npz"), np.load("data/famX_W40.npz")
for n, z in (("famW", w), ("famX", x)):
    m = json.loads(z["meta_json"].item())      # meta_json, NOT meta: savez stores arrays
    print(f"{n}  white_noise={m.get('white_noise', '(pre-flag)')}  "
          f"freq_limit={m.get('freq_limit')}  gains_stored={'kp' in z.files}")
for k in ("kp", "ki", "lhs_samples", "fault_kind"):
    if k in w.files and k in x.files:
        print(f"  {k:12s} bit-equal: {np.array_equal(w[k], x[k])}")
print(f"  Va max |difference|: {np.abs(w['Va'] - x['Va']).max():.5f}  (0.05000 = the noise)")
PY
