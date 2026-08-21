#!/bin/sh
### The two no-fault gain families for hpc/exp16_nofault_gains.txt. ONE job.
###
###     bsub < hpc/job_gen_nofault.sh
###
### Not job_generate.sh: that one hardcodes --W 10 20 40 100 and reads $STEM from the
### environment, and DTU's LSF does not forward the submitting shell's environment
### (the same trap that made 60 sweep jobs exit with "CONFIGS not set"). Everything
### here is written into the script instead, so there is nothing to forget.
###
### TWO separate python invocations, deliberately. Peak RSS is ~8-9 GB during
### _grid_phases + windowing + savez; running both stems in one process risks the
### first family's arrays still being live when the second allocates. Letting the
### interpreter exit in between makes that impossible.
###
### --lhs_seed 11 on BOTH. `create_disturbance_space` returns before it touches the
### RNG when faults are off, so the two families share their initial conditions, grid
### waveforms and gain u-draws exactly -- only the affine map onto (Kp,Ki) differs.
### That is what makes the famM-vs-famL box comparison paired. Do not change one seed
### without changing the other.
###
### WALL TIME 24 h, and do not trim it. The ODE solve is ~7 s; np.savez_compressed on
### ~1 GB per file is the whole cost, and it has now been mis-estimated twice -- 2 h died
### part-way through the fourth file, 4 h was another guess with no measurement behind it.
### The asymmetry is what settles this: asking for more costs a little queue priority,
### while running out costs the elapsed time AND leaves a truncated npz on disk that
### looks like a real dataset. That is exactly how famM_W40 got into a submitted sweep.
### job_sweep.sh already asks 24 h for the same reason.
###
### RESUBMITTING IS SAFE. `gen` below skips a stem whose two files are both already
### there, because generate_family.py's clobber guard would otherwise refuse at the
### FIRST stem and the job would die without ever reaching the one that is missing.
#BSUB -q hpc
#BSUB -J pll_gen_nf
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

# EXISTENCE IS NOT READABILITY. The first attempt died on the 2 h wall limit part-way
# through writing famM_W40.npz, leaving a file that is there and is garbage -- np.load
# raises BadZipFile on it, because a zip's central directory lives at the END and a
# truncated file has none. A skip test that only asked `[ -f ... ]` therefore declared
# the stem done and moved on. It has to open the file.
npz_state () {                                  # -> ok | corrupt | missing | unknown
    python - "$1" <<'PY'
import os, sys, zipfile
import numpy as np
p = sys.argv[1]
if not os.path.exists(p):
    print("missing")
else:
    try:
        np.load(p)                              # reads the zip directory, so truncation fails here
        print("ok")
    except zipfile.BadZipFile:
        print("corrupt")
    except Exception:
        print("unknown")                        # never auto-delete on a reason we do not understand
PY
}

# Regenerating a SUBSET of windowings is safe and gives byte-identical physics:
# generate_multi_W draws the ICs, the gains and the noise ONCE, before the W loop, and
# W only ever drives a reshape. So `--W 40` alone from --lhs_seed 11 reproduces exactly
# the trajectories that produced the good famM_W20 next to it.
gen () {
    stem=$1; shift
    todo=""
    for W in 20 40; do
        f="data/${stem}_W${W}.npz"
        case "$(npz_state "$f")" in
            ok)      ;;
            missing) todo="$todo $W" ;;
            corrupt) echo "removing truncated $f"; rm -f "$f"; todo="$todo $W" ;;
            *)       echo "cannot read $f and it is not a truncated zip -- stopping"; return 1 ;;
        esac
    done
    if [ -z "$todo" ]; then
        echo "skip ${stem}: both windowings present and readable"
        return 0
    fi
    echo "generating ${stem} at W =${todo}"
    # shellcheck disable=SC2086
    python hpc/generate_family.py --stem "$stem" --W $todo --n_runs 5000 \
        --gains --no_faults --lhs_seed 11 "$@"
}

# famL: no faults, the SAME wide gain box as famJ, so famL-vs-famJ moves only the faults
gen famL || exit 1

# famM: no faults, gain box trimmed to the region graphs/rahul/03 measures as accurate
gen famM --kp_range 18 45 --ki_range 180 520 || exit 1

echo "done:"
ls -la data/famL_W*.npz data/famM_W*.npz
