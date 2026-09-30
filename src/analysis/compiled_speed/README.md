# Compiled speed benchmark (F82)

The paper's original 41x / 109x compared the network against a trapezoid solver written as
a Python loop over PyTorch ops, which is mostly interpreter overhead. This folder compares
like with like at batch 1: a compiled (Numba) limited trapezoid solver against a lean,
compiled rollout of the flagship network with the trunk evaluated once.

`numba_solver.py` is the solver from an external review session (2026-09-29); it reproduces
`PLL_Simulator._integrator_step_limited` to 1e-14 rad on three test cases.

Numba does not support the project's Python 3.14, so it runs in its own environment:

    /opt/homebrew/bin/python3.13 -m venv nvenv && nvenv/bin/pip install numba numpy scipy

1. In the PROJECT venv: `.venv/bin/python src/analysis/compiled_speed/export_cases_and_weights.py`
   (writes `cases.npz` and `flagship_weights.npz` here). Then `cd` into this folder for 2 and 3.
2. `nvenv/bin/python check_numba_solver.py`: agreement with the project solver, and its speed.
3. `nvenv/bin/python bench_compiled.py`: lean network (NumPy and Numba) against the compiled
   solver, plus the compiled solver's cost as the step shrinks.
