#!/bin/sh
### The five exp17 families. ONE job.
###
###     bsub < hpc/job_gen_limiter.sh
###
### RESUBMITTING IS SAFE: `gen` skips a stem whose files all OPEN, and deletes and
### regenerates any that are truncated. Existence is not readability -- a job killed
### mid-savez leaves a file that stats fine and raises BadZipFile on load, and a
### stat-only check let exactly that into a submitted sweep on 2026-08-21.
###
### famR carries NO --freq_limit and shares --lhs_seed 21 with famN. The limiter acts
### inside the integrator, after _grid_phases, so the two get bit-identical Va/Vb/Vc --
### a properly paired control for "what did the limiter cost".
###
### DISK: these are ~9 GB together and home is 30 GB. Check `getquota_zhome.sh` first.
### If it is tight, /work3/$USER is large and not backed up, and the code opens datasets
### by relative path -- so generating there and symlinking into data/ is transparent.
#BSUB -q hpc
#BSUB -J pll_gen_lim
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
df -h . | tail -1

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

L=18.8496          # 2*pi*3 rad/s = 3 Hz. NOT 3 rad/s -- see the exp17 header.

gen famN "20 40" --n_runs 5000  --lhs_seed 21 --freq_limit $L          || exit 1
gen famO "20 40" --n_runs 5000  --lhs_seed 22 --freq_limit $L --gains  || exit 1
gen famP "40"    --n_runs 10000 --lhs_seed 23 --freq_limit $L          || exit 1
gen famQ "40"    --n_runs 10000 --lhs_seed 24 --freq_limit $L --gains  || exit 1
gen famR "40"    --n_runs 5000  --lhs_seed 21                          || exit 1

echo "done:"
ls -la data/fam[NOPQR]_W*.npz
df -h . | tail -1
