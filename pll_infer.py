import torch

from PLL_Simulator import PhysicsEquations
from pll_residual import compute_theta_omega
from train_pll import OMEGA_BASE
from train_pll import prepare, load_checkpoint, group_split
from dataset_generator import Dataset_Creator

from omegaconf import OmegaConf
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent / "config"

_phys = PhysicsEquations()
initial_conditions_config = OmegaConf.load(CONFIG_DIR / "initial_conditions.yml")
KP = _phys.Kp
windows = initial_conditions_config.Windows

def interp1d(x_src, y_src, x_query):
    """Linear interpolation of y_src(x_src) onto x_query.
    x_src (S,) ascending, y_src (S,), x_query (T,) -> (T,). Intputs in actual use are sensors, Va or Vb or Vc and the t_query"""
    i = torch.searchsorted(x_src.contiguous(), x_query.contiguous()).clamp(1, x_src.shape[0]-1)  # finds the nearest sensor time at or after the t_query to interpolate 
    lo, hi = i - 1, i
    w = (x_query - x_src[lo]) / (x_src[hi] - x_src[lo])
    return y_src[lo] + w * (y_src[hi] - y_src[lo])


@torch.enable_grad()
def predict_window(model, checkpoint, theta0, omega0, Va, Vb, Vc, times=None):
    """One window. Returns ABSOLUTE theta and omega at `times`.
    theta0, omega0 : scalars    Va,Vb,Vc : (S,)    times : (T,) or None"""
    mu, sd = checkpoint["mu"], checkpoint["sd"]
    t = checkpoint["t_local"] if times is None else times
    T = t.shape[0]

    branch = torch.cat([torch.sin(theta0).view(1, 1), torch.cos(theta0).view(1, 1), ((omega0 - mu) / sd).view(1, 1), Va.view(1, -1), Vb.view(1, -1), Vc.view(1, -1)], dim=-1)
    t_query = t.view(1, T, 1).clone().requires_grad_(True)

    # Vq=0 and omega_nominal=0 make out["omega"] exactly d(theta_dev)/dt, the Kp*Vq term is added back below.
    out = compute_theta_omega(model, t_query, branch, torch.zeros(1, T, 1), omega_nominal=0.0)
    theta_dev = out["theta"].squeeze()
    dtheta_dt = out["omega"].squeeze()
    theta_abs = theta_dev + theta0 + OMEGA_BASE * t  # undo the ramp shift : for the network to learn we remove theta0 and the known 50rad/s ramping

    ts = checkpoint["t_local"]
    same = t.shape == ts.shape and torch.equal(t, ts)
    Va_t = Va if same else interp1d(ts, Va, t)
    Vb_t = Vb if same else interp1d(ts, Vb, t)
    Vc_t = Vc if same else interp1d(ts, Vc, t)
    _, Vq = _phys.park_dqTransform(Va_t, Vb_t, Vc_t, theta_abs)  # Vq isn't available at deployment, we recompute it from the measured, voltages and our angle estimate close the loop.
    omega = dtheta_dt - KP * Vq

    return theta_abs.detach(), omega.detach()


def rollout(model, checkpoint, prep, run_idx, n_windows, W=windows):
    """Chain windows: each window's predicted final state becomes the next IC.
    Va,Vb,Vc are always the true external forcing only the state is fed
    back."""
    t_local = checkpoint["t_local"]
    dt = float(t_local[1] - t_local[0])  # Query one step past the window so the last value lines up with the next window's start
    t_ext = torch.cat([t_local, t_local[-1:] + dt])

    row0 = run_idx * W
    theta0, omega0 = prep["theta0_abs"][row0], prep["omega0"][row0]

    th_all, om_all = [], []
    for k in range(n_windows):
        r = row0 + k
        th, om = predict_window(model, checkpoint, theta0, omega0, prep["Va"][r], prep["Vb"][r], prep["Vc"][r], t_ext)
        th_all.append(th[:-1]); om_all.append(om[:-1])      # drop the extra point
        theta0, omega0 = th[-1], om[-1]                     # feedback
    return torch.cat(th_all), torch.cat(om_all)


def truth(prep, run_idx, n_windows, W=windows):
    row0 = run_idx * W
    return (torch.cat([prep["theta_abs"][row0 + k] for k in range(n_windows)]), torch.cat([prep["target_omega"][row0 + k] for k in range(n_windows)]))

if __name__ == "__main__":
    model, checkpoint = load_checkpoint("pll_deeponet.pth")
    data, meta = Dataset_Creator.load_dataset("pll_dataset.npz")
    prep = prepare(data)
    W = meta["W"]
    tr, va = group_split(prep["run_id"], 0.15, 0)
    val_runs = sorted(set(prep["run_id"][va].tolist()))
    r = val_runs[0]

    #the plot that matters
    for n in [1, 2, 5, 10]:
        thr, _ = rollout(model, checkpoint, prep, r, n, W)
        tht, _ = truth(prep, r, n, W)
        e = thr - tht
        print(f"{n:3d} windows  RMS {torch.sqrt((e**2).mean()):.4e}  max {e.abs().max():.4e} rad")