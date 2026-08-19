#!/bin/sh
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
### RUN TIMES at 4 threads, n_runs=5000, 190 epochs:
###     W=40  2.0 h   |  W=20  1.7 h  |  W=10  2.5 h  |  W=100  2.6 h
### Walltime 8 h is ~3x the longest -- generous, but short enough that LSF can
### backfill these jobs into gaps. A 24 h request would sit in the queue longer.
###
### MEMORY: the benchmark used 1.5 GB but never loads a dataset. Real jobs read a
### ~1 GB npz and hold ~3 GB resident plus the autograd graph (peaks ~5 GB at W=10).
### rusage is PER CORE, so 4 x 4 GB = 16 GB.
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 5GB
#BSUB -W 8:00
##BSUB -u your_email@dtu.dk
#BSUB -o logs/%J_%I.out
#BSUB -e logs/%J_%I.err

cd "$LS_SUBCWD" || exit 1
. .venv/bin/activate

export OMP_NUM_THREADS=${LSB_DJOB_NUMPROC:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTORCH_ENABLE_MPS_FALLBACK=1

CONFIGS="hpc/.expanded/exp2_w_sweep.txt"

LINE=$(sed -n "${LSB_JOBINDEX}p" "$CONFIGS")
[ -n "$LINE" ] || { echo "no config on line $LSB_JOBINDEX of $CONFIGS"; exit 1; }

echo "host $(hostname)  cores $OMP_NUM_THREADS  index $LSB_JOBINDEX"
echo "cmd  python sweep.py $LINE"
# shellcheck disable=SC2086
exec python sweep.py $LINE
