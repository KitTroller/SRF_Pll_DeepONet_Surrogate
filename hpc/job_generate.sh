#!/bin/sh
### Dataset generation -- ONE job, alone. Measured peak RSS ~5.5 GB during
### _grid_phases alone; windowing + savez push it to ~8-9 GB, so 16 GB reserved.
### The ODE solve itself is cheap (1.37 ms/step x 5000 steps = ~7 s at n_runs=5000);
### wall time is dominated by np.savez_compressed on ~1 GB per file.
###
###     export STEM=famB
###     bsub < hpc/job_generate.sh
#BSUB -q hpc
#BSUB -J pll_gen
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -M 3GB
#BSUB -W 1:00
##BSUB -u your_email@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o logs/gen_%J.out
#BSUB -e logs/gen_%J.err

cd "$LS_SUBCWD" || exit 1
. .venv/bin/activate

export OMP_NUM_THREADS=${LSB_DJOB_NUMPROC:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONUNBUFFERED=1

echo "host $(hostname)  cores $OMP_NUM_THREADS  cwd $(pwd)"
python -c "from omegaconf import OmegaConf as O; c=O.load('config/initial_conditions.yml'); print('n_runs =', c.n_runs)"

exec python hpc/generate_family.py --stem "${STEM:-famB}" --W 10 20 40 100
