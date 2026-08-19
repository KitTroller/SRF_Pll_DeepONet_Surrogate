#!/bin/sh
# One-time environment build on DTU HPC. Run from the project root on a LOGIN node:
#
#     sh hpc/setup_env.sh          # CPU-only torch  (hpc queue)
#     sh hpc/setup_env.sh gpu      # CUDA torch      (gpuv100 / gpua100 queues)
#
# requirements.txt is pinned for the laptop (python 3.14): numpy==2.5.2 needs python
# >=3.12 and the newest cluster module may be 3.11. This script loads the newest python
# it can find, tries the exact pins, and falls back to "same packages, latest version
# this python can take" -- then writes hpc/env_lock.txt so the environment every job
# actually used is on the record. All jobs share this one .venv, so they are mutually
# consistent regardless of which branch was taken.
set -eu

MODE=${1:-cpu}

# ---------------------------------------------------------------- python module
echo "available python3 modules:"
module avail python3 2>&1 | sed 's/^/    /' || true

PYOK=""
for cand in python3/3.13.0 python3/3.12.9 python3/3.12.7 python3/3.12.4 \
            python3/3.11.9 python3; do
    module load "$cand" 2>/dev/null || continue
    if python3 -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,10) else 1)' 2>/dev/null; then
        PYOK=$cand; break
    fi
    module unload "$cand" 2>/dev/null || true
done
[ -n "$PYOK" ] || { echo "no usable python3 module. Run 'module avail python3' and add it to the list above."; exit 1; }
echo "using module $PYOK  ->  $(python3 --version)"

if [ "$MODE" = "gpu" ]; then
    module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || \
        echo "WARNING: no cuda module loaded; check 'module avail cuda'"
fi

# ---------------------------------------------------------------- venv
rm -rf .venv
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip wheel >/dev/null
echo "venv python: $(python --version)   pip: $(python -m pip --version | cut -d' ' -f2)"

# ---------------------------------------------------------------- packages
install_relaxed() {
    echo
    echo ">> exact pins do not resolve on $(python --version). Relaxing to latest compatible."
    echo ">> This is fine: every cluster job shares this venv, so all cluster runs stay"
    echo ">> mutually comparable. Do NOT compare them against laptop runs (device trap)."
    sed 's/==.*//' requirements.txt > .req_relaxed
    python -m pip install -r .req_relaxed
    rm -f .req_relaxed
}

if [ "$MODE" = "gpu" ]; then
    python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
    grep -v '^torch' requirements.txt > .req_nogpu
    python -m pip install -r .req_nogpu || { sed 's/==.*//' .req_nogpu > .req_r; python -m pip install -r .req_r; rm -f .req_r; }
    rm -f .req_nogpu
else
    python -m pip install -r requirements.txt || install_relaxed
fi

mkdir -p logs runs
python -m pip freeze > hpc/env_lock.txt
echo
echo "resolved versions (also written to hpc/env_lock.txt):"
grep -iE '^(torch|numpy|scipy|matplotlib|omegaconf|pytorch[-_]optimizer)' hpc/env_lock.txt | sed 's/^/    /'

# ---------------------------------------------------------------- smoke test
echo
python hpc/smoke_test.py
echo
echo "OK. Next:  STEM=famB bsub < hpc/job_generate.sh"
