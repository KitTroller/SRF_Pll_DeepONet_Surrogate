#!/bin/sh
### 48-HOUR variant, on the `milan` queue. For the cells that do not fit 24 h at all.
###
### WHY A DIFFERENT QUEUE: `hpc` caps at 24 h (1440 min) and so do `epyc` and `rome`.
### `milan` allows 2880 min = 48 h, and at the time of writing had 0 jobs pending against
### 3542 pending on `hpc`. Both limits confirmed with `bqueues -l <queue>`.
###
### WHY IT IS NEEDED, measured rather than guessed. TERM_RUNLIMIT killed 11 of 12 cells at
### exactly 86409 s with CPU/wall = 7.99, so all 8 cores were working and it still was not
### enough. The reason the earlier 17 h estimate was wrong: bigger networks do not only
### cost more per epoch, they run MORE EPOCHS before early-stopping --
###
###     L2_w32   20,896 params   ~650 epochs    5-7 h
###     L3_w64   54,016          ~750-880      10-11 h
###     L2_w128 107,584          ~720-1050     19-20 h
###     L4_w128 173,632          ~990-1120     20 h+
###
### 3.8x the parameters AND 1.6x the epochs. I scaled the first and assumed the second
### held; it does not.
###
### MILAN IS AMD, the rest of this project was measured on Xeon Gold 6226R. Minibatch
### order is unaffected (`torch.randperm` on device=cpu either way), but BLAS kernels
### differ, so expect tiny numerical differences. Against a 1.6x seed spread and the 2x
### effects being measured, that is noise -- but note which cells ran where if a result
### ever hinges on a 5% difference.
###
### 8-CORE variant of job_sweep.sh, for the arms that ran out of wall time.
###
### exp17's width=128 network is 107,584 params against the baseline's 45,696 -- roughly
### 2.4x the compute per epoch, so ~67 s/epoch instead of ~28, and 1200 epochs is ~22 h.
### With queue contention that exceeded the 24 h limit and LSF killed those elements with
### SIGUSR2 ("User defined signal 2"). The 24 h in job_sweep.sh was sized for the BASELINE
### network and never re-checked when wider arms were added.
###
### 8 threads measured 76 ms/step against 107 at 4 (see below), so ~1.4x -- enough to
### bring the w128 arms inside the limit. Do not use this for the ordinary arms: 8 cores
### at 55% efficiency is a worse use of the allocation than 4 at 78%.
###
### One sweep config per array element. Submitted by hpc/submit.sh, which sets
### $CONFIGS and the [1-N] range so they cannot drift apart.
###
### Sizing (measured on the laptop, batch_size=512, F=4):
###   step cost   W=100 39 ms | W=40 77 ms | W=20 119 ms | W=10 190 ms   (CPU, 4 threads)
###   at n_runs=5000 that is ~35 / 28 / 21 / 17 s per epoch, ~200 epochs to early stop
###   -> longest run (W=100) ~2.5 h. 8 h gives room for a slow node.
###
### CORES: 4. Measured on the real node (Xeon Gold 6226R), ms/step at W=40:
###     1 thread 334  |  2 threads 187  |  4 threads 107  |  8 threads 76
### That is 3.1x from 4 cores (78% efficiency) and 4.4x from 8 (55%). The M1 Max
### scaled far worse, which is why the earlier -n 2 guess was wrong. 4 cores is the
### balance: ~2 h per job, and 60 jobs x 4 cores is a plausible concurrent allocation
### where 60 x 8 would queue in waves.
###
### WALLTIME 24 h, and here is why the first attempt died at 8 h.
### The benchmark predicted W=40 ~2.0 h, W=100 ~2.6 h at 4 threads. The real jobs were
### still running at 7.6 h and 34 of 60 were killed by the 8 h limit. Two reasons the
### benchmark under-predicted, both my error:
###   1. It reused fixed tensors. The real epoch does `t[idx]` gathers over a
###      200000 x 378 branch tensor -- memory-bandwidth bound and barely threaded.
###      Measured effective parallelism on a real job: 52703 s CPU / 27400 s wall
###      = 1.9 cores, not 4.
###   2. Runs do not early-stop as fast as Stage C did on MPS (178 epochs); with
###      epochs=800/patience=40 some go far longer.
### Realistic: ~2x the benchmark, so 4-6 h per job. 24 h is the safe request; the
### queue penalty for asking is smaller than losing 8 h of compute.
#BSUB -q milan
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 5GB
#BSUB -W 47:00
##BSUB -u your_email@dtu.dk
#BSUB -o logs/%J_%I.out
#BSUB -e logs/%J_%I.err

cd "$LS_SUBCWD" || exit 1
. .venv/bin/activate

export OMP_NUM_THREADS=${LSB_DJOB_NUMPROC:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

# Python block-buffers stdout when it is not a TTY, so a 2-hour job shows nothing in
# its .out file until the buffer fills. Unbuffered costs nothing here and makes
# `tail -f logs/<jobid>_<index>.out` actually work.
export PYTHONUNBUFFERED=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

: "${CONFIGS:?CONFIGS not set -- submit via hpc/submit.sh, not bare bsub}"

LINE=$(sed -n "${LSB_JOBINDEX}p" "$CONFIGS")
[ -n "$LINE" ] || { echo "no config on line $LSB_JOBINDEX of $CONFIGS"; exit 1; }

echo "host $(hostname)  cores $OMP_NUM_THREADS  index $LSB_JOBINDEX"
echo "cmd  python src/sweep.py $LINE"
# shellcheck disable=SC2086
exec python src/sweep.py $LINE
