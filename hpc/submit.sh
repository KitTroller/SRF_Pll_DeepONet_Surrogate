#!/bin/sh
# Submit one LSF array job, one element per config line.
#
#     sh hpc/submit.sh hpc/exp1_w40_fourier.txt
#     sh hpc/submit.sh hpc/exp2_w_sweep.txt  wsweep  hpc/job_sweep_gpu.sh
#
# Comments and blank lines are stripped into hpc/.expanded/, and the array range is
# taken from THAT file, so the [1-N] range and the line numbers sed reads can never
# disagree.
#
# The config path is BAKED INTO a generated copy of the jobscript rather than passed
# as an environment variable. DTU's LSF does not forward the submitting shell's
# environment to the job -- an earlier version relied on that and all 60 jobs exited
# immediately with "CONFIGS not set". No env, no problem.
set -eu

SRC=${1:?usage: hpc/submit.sh <configs.txt> [jobname] [jobscript]}
NAME=${2:-pllsweep}
SCRIPT=${3:-hpc/job_sweep.sh}

[ -f "$SRC" ]    || { echo "no such config file: $SRC"; exit 1; }
[ -f "$SCRIPT" ] || { echo "no such jobscript: $SCRIPT"; exit 1; }

mkdir -p logs hpc/.expanded
EXP="hpc/.expanded/$(basename "$SRC")"
sed -e 's/[[:space:]]*#.*$//' -e '/^[[:space:]]*$/d' "$SRC" > "$EXP"

N=$(wc -l < "$EXP" | tr -d ' ')
[ "$N" -gt 0 ] || { echo "no configs in $SRC"; exit 1; }

# every dataset named in the configs must already exist, or the array dies 48 times over
missing=$(awk '{for(i=1;i<=NF;i++) if($i=="--dataset") print $(i+1)}' "$EXP" | sort -u \
          | while read -r d; do [ -f "$d" ] || [ -f "data/$d" ] || echo "$d"; done)
[ -z "$missing" ] || { echo "missing dataset(s), refusing to submit:"; echo "$missing" | sed 's/^/    /'; exit 1; }

# generated jobscript with the config path written in
JOB="hpc/.expanded/$(basename "$SRC" .txt)__$(basename "$SCRIPT")"
sed "s|^: \"\\\${CONFIGS:?.*|CONFIGS=\"$EXP\"|" "$SCRIPT" > "$JOB"
grep -q "^CONFIGS=" "$JOB" || { echo "could not inject CONFIGS into $SCRIPT -- did its ': \${CONFIGS:?...}' line change?"; exit 1; }

echo "$SRC -> $N jobs via $SCRIPT   (config path baked into $JOB)"
nl -ba "$EXP"
echo

bsub -J "${NAME}[1-${N}]" < "$JOB"

echo
echo "watch with:  bstat              (or bjobs -A for the array summary)"
echo "kill  with:  bkill -J ${NAME}"
echo "logs:        logs/<jobid>_<index>.out   (.err for errors)"
