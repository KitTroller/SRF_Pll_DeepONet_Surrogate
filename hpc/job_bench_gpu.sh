#!/bin/sh
### Same benchmark on a V100. The venv already has torch+cu130, nothing to reinstall.
### Runs the GPU pass AND a cpu pass on the same node, so the comparison is on
### identical hardware rather than across two different queues.
###
###     bsub < hpc/job_bench_gpu.sh    then: cat logs/benchgpu_<jobid>.out
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J pll_benchgpu
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 5GB
#BSUB -W 0:30
#BSUB -o logs/benchgpu_%J.out
#BSUB -e logs/benchgpu_%J.err

cd "$LS_SUBCWD" || exit 1
. .venv/bin/activate

echo "host $(hostname)"
grep -m1 'model name' /proc/cpuinfo 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "NO GPU VISIBLE"
echo
echo "================ GPU ================"
python hpc/bench.py cuda
echo
echo "================ CPU on the same node ================"
python hpc/bench.py cpu
