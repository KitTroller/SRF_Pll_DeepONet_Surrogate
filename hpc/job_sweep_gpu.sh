#!/bin/sh
### GPU variant. Same contract as job_sweep.sh -- submitted by hpc/submit.sh.
###
### Worth knowing before you choose this queue:
###  - the model is tiny (45k params). The step is latency-bound on small kernels,
###    not throughput-bound, so a V100 will NOT give you an order of magnitude.
###    Measure it: submit ONE config here and one on the hpc queue, compare `seconds`
###    in the two JSON records.
###  - the rollout evaluation at the end of main() runs on CPU regardless
###    (load_checkpoint defaults to device="cpu") and is batch-size-1, ~12k calls
###    at W=40/n_eval=150. That is GPU time spent idle.
###  - the hpc queue has far more slots, so for a 15-job array it usually wins on
###    wall-clock even at ~3.5x slower per run.
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 5GB
#BSUB -W 24:00
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

: "${CONFIGS:?CONFIGS not set -- submit via hpc/submit.sh, not bare bsub}"

LINE=$(sed -n "${LSB_JOBINDEX}p" "$CONFIGS")
[ -n "$LINE" ] || { echo "no config on line $LSB_JOBINDEX of $CONFIGS"; exit 1; }

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "cmd  python src/sweep.py $LINE --device cuda"
# shellcheck disable=SC2086
exec python src/sweep.py $LINE --device cuda
