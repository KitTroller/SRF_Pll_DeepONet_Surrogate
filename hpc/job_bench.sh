#!/bin/sh
### Reality check on a COMPUTE node. Login-node timings are useless -- DTU pins
### login shells to 1 thread. Everything runs in ONE python process and any error
### is printed in full, so a failure still tells us something.
###
###     bsub < hpc/job_bench.sh    then: cat logs/bench_<jobid>.out
#BSUB -q hpc
#BSUB -J pll_bench
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -M 3GB
#BSUB -W 0:30
#BSUB -o logs/bench_%J.out
#BSUB -e logs/bench_%J.err

cd "$LS_SUBCWD" || exit 1
. .venv/bin/activate

echo "host $(hostname)"
grep -m1 'model name' /proc/cpuinfo 2>/dev/null || true
echo
exec python hpc/bench.py cpu
