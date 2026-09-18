#!/bin/sh
### GPU SMOKE TEST. One job, 10 epochs of the deliverable config on an L40S. Answers two
### questions before any experiment is sent to a GPU queue:
###   1. does training run on a GPU here at all? (the V100 never did -- no Volta kernels
###      in the cu130 torch build, see job_sweep_gpu.sh)
###   2. how many s/epoch? Read `seconds / epochs_run` in the record and compare with
###      milan's ~127 s/epoch for the same config at 8 cores (exp28 baseline seeds 0-3).
###
###     bsub < hpc/job_gpu_smoke.sh      then: cat logs/gpusmoke_<jobid>.out
###
### SAFE BY CONSTRUCTION. --seed 90 gives a tag no real run uses, so it cannot overwrite a
### deliverable checkpoint in runs/ (sweep.py has no --runs_dir). Its record goes to
### sweeps_gpu_smoke/, which no analysis reads. The 10-epoch model is useless; delete
### runs/*_s90sp0_* afterwards.
###
### --device cuda is explicit: train_pll.DEVICE auto-detects, so without it a node where
### torch cannot see the GPU would silently train on CPU and this would time the CPU.
###
### 1 h wall so it can backfill before a service window. If 10 epochs do not fit in an
### hour, that is the answer too: the GPU is slower than milan.
#BSUB -q gpul40s
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J pll_gpusmoke
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 5GB
#BSUB -W 1:00
#BSUB -o logs/gpusmoke_%J.out
#BSUB -e logs/gpusmoke_%J.err

cd "$LS_SUBCWD" || exit 1
. .venv/bin/activate

export OMP_NUM_THREADS=${LSB_DJOB_NUMPROC:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONUNBUFFERED=1

echo "host $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "NO GPU VISIBLE"
python -c "import torch; ok = torch.cuda.is_available(); print('torch', torch.__version__, '| cuda', ok, '|', torch.cuda.get_device_name() if ok else '', torch.cuda.get_device_capability() if ok else '')"

exec python src/sweep.py --dataset famO_W40.npz --F 4 --max_freq 503 --w_phys 0.3 \
    --split_seed 0 --epochs 10 --patience 40 --n_eval_runs 150 --n_layers 3 --width 128 \
    --results_dir sweeps_gpu_smoke --seed 90 --device cuda
