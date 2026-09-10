#!/bin/sh
### 16-CORE variant of job_sweep_long.sh, for the L4_w128 cells that do not fit 47 h.
###
###     sh hpc/submit.sh hpc/exp25_frontier_missing.txt front16 hpc/job_sweep_long16.sh
###
### WHY MORE CORES AND NOT MORE HOURS. `milan` allows 2880 min = 48 h and job_sweep_long.sh
### already asks 47:00, so there is one hour of headroom left on the whole queue -- there
### is nowhere to go on time. Measured from the runs that DID finish (`seconds` in their
### sweep records):
###
###     L4_w128 completed:  38.6, 38.9, 39.1, 43.3, 43.4, 43.7 h   at 922-1028 epochs
###
### The worst survivor used 43.7 h of a 47 h cap. That is 7% of headroom, and a cell that
### early-stops at 1120 epochs instead of 1004 needs ~49 h. frontlong[3] and frontlong[11]
### did not get unlucky on a slow node; 47 h simply is not enough for the tail of the
### epoch distribution. Asking for 8 more cores buys far more than asking for 1 more hour.
###
### THE SCALING IS REAL, not assumed. The cells killed at the 24 h cap reported
### CPU/wall = 7.99 on 8 cores, i.e. all 8 were saturated, so this workload does use what
### it is given. Do NOT expect a clean 2x: `t[idx]` gathers over a 200000 x 378 branch
### tensor are memory-bandwidth bound and thread poorly -- an earlier 4-core job measured
### only 1.9 effective cores on that phase. Budget ~0.6-0.7x wall, which puts the worst
### case near 30 h with 17 h of margin.
###
### NOTHING NUMERICAL CHANGES. Minibatch order is `torch.randperm` on device=cpu and the
### seeds are unchanged, so these runs stay comparable to the 8-core ones. Only the BLAS
### thread count differs, the same class of difference as milan-vs-Xeon already noted in
### job_sweep_long.sh. Note which cells ran where if a result ever turns on 5%.
###
### The queue wait may be longer for a 16-core slot than an 8-core one. For three jobs
### that trade is obviously worth it; do not reach for this script for a 100-job array
### without checking `bqueues -l milan` first.
#BSUB -q milan
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 5GB
#BSUB -W 47:00
#BSUB -o logs/%J_%I.out
#BSUB -e logs/%J_%I.err

cd "$LS_SUBCWD" || exit 1
. .venv/bin/activate

export OMP_NUM_THREADS=${LSB_DJOB_NUMPROC:-16}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

# Unbuffered, or a 30-hour job shows nothing in its .out until the buffer fills.
export PYTHONUNBUFFERED=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

: "${CONFIGS:?CONFIGS not set -- submit via hpc/submit.sh, not bare bsub}"

LINE=$(sed -n "${LSB_JOBINDEX}p" "$CONFIGS")
[ -n "$LINE" ] || { echo "no config on line $LSB_JOBINDEX of $CONFIGS"; exit 1; }

echo "host $(hostname)  cores $OMP_NUM_THREADS  index $LSB_JOBINDEX"
echo "cmd  python src/sweep.py $LINE"
# shellcheck disable=SC2086
exec python src/sweep.py $LINE
