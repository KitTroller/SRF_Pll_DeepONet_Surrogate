#!/bin/sh
### LONG GPU variant (16 h, 8 GB/core) for n=40000. Same contract as job_sweep.sh -- submitted by hpc/submit.sh.
###
### Worth knowing before you choose this queue:
###  - NOT gpuv100. The cluster .venv is torch 2.13.0+cu130, and CUDA 13 ships no Volta
###    kernels: the V100 benchmark died on its first tensor with "no kernel image is
###    available for execution on the device" (logs/benchgpu_29137820.err). So the GPU
###    was never measured before 2026-09-18 -- the old "a V100 will not give an order
###    of magnitude" was a FLOP estimate for the 45k-param model, not a result.
###    gpul40s (L40S, 48 h) is used instead; its "no double precision" does not matter,
###    training is float32 (train_pll.py:19). gpua100 (72 h) also works but queues deep.
###  - MEASURED 2026-09-18 (hpc/job_gpu_smoke.sh): famO L3_w128 trains at
###    7.95 s/epoch on an L40S against ~127 s/epoch on 8 milan cores -- 16x, and that 10-epoch
###    figure includes CUDA start-up. The old "this workload cannot use a big GPU" was a FLOP
###    estimate for the 45k-param model at batch 1; at batch 512 x 125 time points it can.
###  - THIS FILE IS FOR n=40000 ONLY (exp33): -W 16:00 and 8 GB/core. ~35 s/epoch by the
###    measured scaling (5 s at n=5000, 9 s at 10000, 18 s at 20000), so the 1200-epoch worst
###    case is ~11.7 h; the 8 GB/core covers the slim file's float64 rebuild at load (~11 GB)
###    plus the training tensors. Use hpc/job_sweep_gpu.sh (10 h, 4 GB/core) for smaller runs.
###  - sizing below is the 10 h script's, kept for reference: WORST case, 1200 epochs, from the
###    epoch times the real runs measured (exp29-31: famO 4.3-5.3 s, famQ 8.6-9.2 s):
###      famO 1200 x 5 s = 1.7 h    famQ 1200 x 9 s = 3.0 h    famO20k 1200 x ~18 s = 6.0 h
###    A 47 h request only hurts: LSF backfills short jobs first, and a service window
###    blocks any job whose limit would cross it. Raise it for anything bigger than famQ.
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
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 9GB
#BSUB -W 16:00
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
