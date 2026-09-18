#!/bin/sh
### GPU variant. Same contract as job_sweep.sh -- submitted by hpc/submit.sh.
###
### Worth knowing before you choose this queue:
###  - NOT gpuv100. The cluster .venv is torch 2.13.0+cu130, and CUDA 13 ships no Volta
###    kernels: the V100 benchmark died on its first tensor with "no kernel image is
###    available for execution on the device" (logs/benchgpu_29137820.err). So the GPU
###    has NEVER been measured on this project -- the old "a V100 will not give an order
###    of magnitude" was a FLOP estimate for the 45k-param model, not a result.
###    gpul40s (L40S, 48 h) is used instead; its "no double precision" does not matter,
###    training is float32 (train_pll.py:19). gpua100 (72 h) also works but queues deep.
###  - run hpc/job_gpu_smoke.sh FIRST: 10 epochs, gives s/epoch against milan's ~127
###    s/epoch (famO L3_w128, 8 cores). Do not send an experiment here before that.
###  - the rollout evaluation at the end of main() runs on CPU regardless
###    (load_checkpoint defaults to device="cpu") and is batch-size-1, ~12k calls
###    at W=40/n_eval=150. That is GPU time spent idle.
###  - GPU queues do NOT count against milan's 96-slot cap, so a GPU array runs
###    alongside a milan array rather than instead of it.
###  - a GPU run is a fresh draw, not a replica of the same seed on CPU (different
###    kernels AND a different randperm stream). Never compare one GPU seed to one CPU seed.
#BSUB -q gpul40s
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
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

: "${CONFIGS:?CONFIGS not set -- submit via hpc/submit.sh, not bare bsub}"

LINE=$(sed -n "${LSB_JOBINDEX}p" "$CONFIGS")
[ -n "$LINE" ] || { echo "no config on line $LSB_JOBINDEX of $CONFIGS"; exit 1; }

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "cmd  python src/sweep.py $LINE --device cuda"
# shellcheck disable=SC2086
exec python src/sweep.py $LINE --device cuda
