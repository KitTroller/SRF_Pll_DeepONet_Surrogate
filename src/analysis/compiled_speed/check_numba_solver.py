import numpy as np, time
from numba_solver import run
d = np.load("cases.npz")
dt, Kp, Ki, L, b = 100e-6, 25.0, 300.0, 18.8496, 0.05
for s in range(3):
    th, it = run(d[f"Va{s}"], d[f"Vb{s}"], d[f"Vc{s}"], float(d[f"th0_{s}"]), float(d[f"om0_{s}"]), dt, Kp, Ki, L, b)
    err = np.abs(th - d[f"th{s}"]).max()
    ts = []
    for _ in range(30):
        t0 = time.perf_counter(); run(d[f"Va{s}"], d[f"Vb{s}"], d[f"Vc{s}"], float(d[f"th0_{s}"]), float(d[f"om0_{s}"]), dt, Kp, Ki, L, b); ts.append(time.perf_counter() - t0)
    print(f"case {s}: max|theta_numba - theta_torch| = {err:.2e} rad | {np.median(ts)*2e3:.2f} ms per simulated second | Newton iters/step {it/(len(th)-1):.2f}")
