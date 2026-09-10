#!/bin/sh
### 24-CORE variant, for the width=256 cells. `milan`, 47 h.
###
###     sh hpc/submit.sh hpc/exp26_w256.txt w256 hpc/job_sweep_w256.sh
###
### WHY 24 CORES AND NOT 16. Sized from measured s/epoch, not guessed:
###
###     L2_w64   44.8 s/epoch    L2_w128   68.4   -> 1.53x per width doubling
###     L3_w64   49.6            L3_w128   93.2   -> 1.88x
###
### The factor grows with width because the matmul share grows, so w128 -> w256 should be
### worse than either: budget ~1.9x for L2 and ~2.4x for L3, i.e. ~130 and ~224 s/epoch.
### At the ~900 epochs the w128 cells needed that is 32 h and 56 h AT 8 CORES. L3_w256
### does not fit 47 h at 8 cores at all, and at 16 cores it lands near 36 h -- which is
### the same thin margin that killed frontlong[3] and [11], because the binding case is
### not the median epoch count but the tail: a cell that runs to the 1200 cap needs
### 75 h at 8 cores.
###
### THE 8->16->24 CORE FACTOR IS A BUDGET, NOT A MEASUREMENT. The only hard number is
### CPU/wall = 7.99 on 8 cores, which says the work saturates what it is given but not how
### it scales past that; the `t[idx]` gathers over a 200000 x 378 branch tensor are
### memory-bandwidth bound and an old 4-core job measured only 1.9 effective cores on that
### phase. exp25 is running L4_w128 at 16 cores right now and its 8-core time is known
### (41.8 h), so it will MEASURE the factor. If it comes back much worse than 0.65x, come
### back here before submitting anything larger than w256.
###
### CHECK THE SLOT EXISTS. If these pend forever, milan's nodes may not offer 24 cores on
### one host:  bqueues -l milan   and   lshosts -w | head
### Dropping to -n 16 costs ~1.5x wall and still fits L2_w256 comfortably.
###
### NOTHING NUMERICAL CHANGES with core count: minibatch order is `torch.randperm` on
### device=cpu with unchanged seeds. Only BLAS threading differs -- same class of caveat
### as milan-vs-Xeon in job_sweep_long.sh.
#BSUB -q milan
#BSUB -n 24
#BSUB -R "span[hosts=1]"
# 4/5, not 6/8: DTU's esub rejects a hard limit more than 1 GB above the reservation
# ("The memory limit must not exceed the usage by more than 1GB!"). Memory here is
# dominated by the shared 200000 x 378 branch tensor, not by the parameters, so w256
# needs no more than the w128 cells that ran fine at these numbers.
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 5GB
#BSUB -W 47:00
#BSUB -o logs/%J_%I.out
#BSUB -e logs/%J_%I.err

cd "$LS_SUBCWD" || exit 1
. .venv/bin/activate

export OMP_NUM_THREADS=${LSB_DJOB_NUMPROC:-24}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

export PYTHONUNBUFFERED=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

: "${CONFIGS:?CONFIGS not set -- submit via hpc/submit.sh, not bare bsub}"

LINE=$(sed -n "${LSB_JOBINDEX}p" "$CONFIGS")
[ -n "$LINE" ] || { echo "no config on line $LSB_JOBINDEX of $CONFIGS"; exit 1; }

echo "host $(hostname)  cores $OMP_NUM_THREADS  index $LSB_JOBINDEX"
echo "cmd  python src/sweep.py $LINE"
# shellcheck disable=SC2086
exec python src/sweep.py $LINE
