import numpy as np, time
from numba import njit
from numba_solver import run as solver
d, c = np.load("flagship_weights.npz"), np.load("cases.npz")
mu, sd, kmu, ksd, imu, isd, wb = d["norm"]; t_ext = d["t_ext"]
x = d["trunk_in"]
for i in range(4):                                   # trunk evaluated ONCE: it depends only on t
    x = x @ d[f"tW{i}"].T + d[f"tb{i}"]
    x = np.tanh(x) if i < 3 else x
TT = np.ascontiguousarray(x.T)                       # (64, 126) basis
bW = [np.ascontiguousarray(d[f"bW{i}"]) for i in range(4)]; bb = [d[f"bb{i}"].copy() for i in range(4)]
kpn, kin = (25 - kmu) / ksd, (300 - imu) / isd

def lean_np(Va, Vb, Vc, th0, om0):
    out, z = np.empty(5000), np.empty(380)
    for w in range(40):
        s = slice(w * 125, (w + 1) * 125)
        z[0], z[1], z[2] = np.sin(th0), np.cos(th0), (om0 - mu) / sd
        z[3:128], z[128:253], z[253:378], z[378], z[379] = Va[s], Vb[s], Vc[s], kpn, kin
        h = np.tanh(bW[0] @ z + bb[0]); h = np.tanh(bW[1] @ h + bb[1]); h = np.tanh(bW[2] @ h + bb[2])
        N = (bW[3] @ h + bb[3]).reshape(2, 64) @ TT
        th = N[0] + th0 + wb * t_ext; om = om0 + N[1] - N[1, 0]      # om: the anchor
        out[s] = th[:125]; th0, om0 = th[125], om[125]
    return out

@njit(fastmath=False)
def lean_nb(Va, Vb, Vc, th0, om0, W0, b0, W1, b1, W2, b2, W3, b3, TT, mu, sd, kpn, kin, wb, t_ext):
    out, z = np.empty(5000), np.empty(380)
    for w in range(40):
        o = w * 125
        z[0] = np.sin(th0); z[1] = np.cos(th0); z[2] = (om0 - mu) / sd
        z[3:128] = Va[o:o + 125]; z[128:253] = Vb[o:o + 125]; z[253:378] = Vc[o:o + 125]; z[378] = kpn; z[379] = kin
        h = np.tanh(W0 @ z + b0); h = np.tanh(W1 @ h + b1); h = np.tanh(W2 @ h + b2)
        N = (W3 @ h + b3).reshape(2, 64) @ TT
        th = N[0] + th0 + wb * t_ext; om = om0 + N[1] - N[1, 0]
        out[o:o + 125] = th[:125]; th0 = th[125]; om0 = om[125]
    return out

args = lambda s: (c[f"Va{s}"], c[f"Vb{s}"], c[f"Vc{s}"], float(c[f"th0_{s}"]), float(c[f"om0_{s}"]))
nbx = (bW[0], bb[0], bW[1], bb[1], bW[2], bb[2], bW[3], bb[3], TT, mu, sd, kpn, kin, wb, t_ext)
ref = d["ref_theta"]
print(f"lean NumPy vs deployed torch rollout: max |dtheta| = {np.abs(lean_np(*args(0)) - ref).max():.1e} rad")
print(f"lean Numba vs deployed torch rollout: max |dtheta| = {np.abs(lean_nb(*args(0), *nbx) - ref).max():.1e} rad")
def med(f, n=30):
    f(); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return np.median(ts) * 2e3                       # 0.5 s run -> ms per simulated second
print(f"\nbatch 1, ms per simulated second (M1 Max, one process, nothing else running):")
print(f"  compiled trapezoid solver, limited, 100 us : {med(lambda: solver(*args(0), 100e-6, 25.0, 300.0, 18.8496, 0.05)):6.2f}")
print(f"  lean network, NumPy                        : {med(lambda: lean_np(*args(0))):6.2f}")
print(f"  lean network, Numba                        : {med(lambda: lean_nb(*args(0), *nbx)):6.2f}")
print("\ncompiled solver vs its step (the network's cost does not depend on the solver step):")
for dt in (100e-6, 50e-6, 25e-6, 10e-6, 5e-6, 2e-6):
    n = int(round(0.5 / dt)); t = np.arange(n) * dt; w = 2 * np.pi * 50; r = np.random.default_rng(0)
    V = [np.cos(w * t + p) + 0.1 * (r.random(n) - .5) for p in (0, -2 * np.pi / 3, 2 * np.pi / 3)]
    print(f"  dt = {dt*1e6:5.0f} us : {med(lambda: solver(*V, 0.3, 5.0, dt, 25.0, 300.0, 18.8496, 0.05), n=7):8.2f} ms per simulated second")
