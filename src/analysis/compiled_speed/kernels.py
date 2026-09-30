"""Numba kernels shared by bench_head_to_head.py and bench_scaling.py."""
import numpy as np
from numba import njit

W0, TWO3 = 2 * np.pi * 50, 2 * np.pi / 3


@njit
def park_q(Va, Vb, Vc, th):
    return -(2 / 3) * (Va * np.sin(th) + Vb * np.sin(th - TWO3) + Vc * np.sin(th + TWO3))


@njit
def park_d(Va, Vb, Vc, th):
    return (2 / 3) * (Va * np.cos(th) + Vb * np.cos(th - TWO3) + Vc * np.cos(th + TWO3))


@njit
def solver(Va, Vb, Vc, th0, om0, dt, Kp, Ki):
    n = Va.shape[0]; th = np.empty(n); th[0] = th0; om = om0
    vq_p = park_q(Va[0], Vb[0], Vc[0], th0); b = (dt / 2) * (Kp + (dt / 2) * Ki)
    for k in range(1, n):
        known = th[k - 1] + dt * (om + W0) + b * vq_p; t = known
        for _ in range(10):
            s = (t - known - b * park_q(Va[k], Vb[k], Vc[k], t)) / (1.0 + b * park_d(Va[k], Vb[k], Vc[k], t))
            t -= s
            if abs(s) < 1e-10:
                break
        vq = park_q(Va[k], Vb[k], Vc[k], t)
        om = om + dt / 2 * Ki * (vq_p + vq); th[k] = t; vq_p = vq
    return th


@njit
def theirs(Va, Vb, Vc, th0, om0, dt, Ki, W1e, B1e, W2, B2):
    n = Va.shape[0]; out = np.empty(n); x = np.empty(5)
    va = (2 / 3) * Va - (1 / 3) * Vb - (1 / 3) * Vc; vb = (Vb - Vc) / np.sqrt(3.0)
    x1 = om0 / Ki; th = th0 % (2 * np.pi); vq = -np.sin(th) * va[0] + np.cos(th) * vb[0]; out[0] = th
    for j in range(1, n):
        x[0] = x1; x[1] = th; x[2] = vq; x[3] = va[j]; x[4] = vb[j]
        p = W2 @ np.tanh(W1e @ x + B1e) + B2
        x1 = p[0] * dt + x1; th = (p[1] * dt + th) % (2 * np.pi)
        vq = -np.sin(th) * va[j] + np.cos(th) * vb[j]; out[j] = th
    return out


@njit
def ours(Va, Vb, Vc, th0, om0, bW0, bb0, bW1, bb1, bW2, bb2, bW3, bb3, TT, mu, sd, t_ext):
    out = np.empty(5000); z = np.empty(378)
    for w in range(40):
        o = w * 125
        z[0] = np.sin(th0); z[1] = np.cos(th0); z[2] = (om0 - mu) / sd
        z[3:128] = Va[o:o + 125]; z[128:253] = Vb[o:o + 125]; z[253:378] = Vc[o:o + 125]
        h = np.tanh(bW0 @ z + bb0); h = np.tanh(bW1 @ h + bb1); h = np.tanh(bW2 @ h + bb2)
        N = (bW3 @ h + bb3).reshape(2, 64) @ TT
        th = N[0] + th0 + W0 * t_ext
        out[o:o + 125] = th[:125]; th0 = th[125]; om0 = N[1, 125]        # no anchor on famR
    return out


@njit
def mlp(Va, Vb, Vc, th0, om0, W1b, T1, W2t, b2, W3t, b3, mu, sd, t_ext):
    out = np.empty(5000); z = np.empty(378)
    for w in range(40):
        o = w * 125
        z[0] = np.sin(th0); z[1] = np.cos(th0); z[2] = (om0 - mu) / sd
        z[3:128] = Va[o:o + 125]; z[128:253] = Vb[o:o + 125]; z[253:378] = Vc[o:o + 125]
        h1 = np.tanh(T1 + W1b @ z)                       # (126, 64): branch part broadcast over t
        y = np.tanh(h1 @ W2t + b2) @ W3t + b3            # (126, 2)
        th = y[:, 0] + th0 + W0 * t_ext
        out[o:o + 125] = th[:125]; th0 = th[125]; om0 = y[125, 1]
    return out




@njit
def ours_fine(Va, Vb, Vc, k, th0, om0, bW0, bb0, bW1, bb1, bW2, bb2, bW3, bb3, TTf, mu, sd, kpn, kin, wb, t_fine):
    """The flagship (anchor trained in) emitting theta at EVERY simulator step. The branch still
    sees 125 samples per window (every k-th input sample); the output product grows with the
    number of output points m = 12.5 ms / dt, which is the honest cost of fine-step output."""
    m = TTf.shape[1] - 1
    nwin = Va.shape[0] // m
    out = np.empty(nwin * m); z = np.empty(380)
    for w in range(nwin):
        o = w * m
        z[0] = np.sin(th0); z[1] = np.cos(th0); z[2] = (om0 - mu) / sd
        for i in range(125):
            z[3 + i] = Va[o + i * k]; z[128 + i] = Vb[o + i * k]; z[253 + i] = Vc[o + i * k]
        z[378] = kpn; z[379] = kin
        h = np.tanh(bW0 @ z + bb0); h = np.tanh(bW1 @ h + bb1); h = np.tanh(bW2 @ h + bb2)
        b = bW3 @ h + bb3
        th = b[:64] @ TTf + th0 + wb * t_fine
        om0 = om0 + b[64:] @ TTf[:, m] - b[64:] @ TTf[:, 0]          # the anchor, at t = T_w
        out[o:o + m] = th[:m]; th0 = th[m]
    return out
