import torch

from PLL_Simulator import PhysicsEquations
from pll_residual import compute_theta_omega, build_trunk_input
from train_pll import OMEGA_BASE
from train_pll import prepare, load_checkpoint, group_split
from dataset_generator import Dataset_Creator

from omegaconf import OmegaConf
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"   # src/ -> root

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


def predict_window(model, checkpoint, theta0, omega0, Va, Vb, Vc, times=None):
    mu, sd = checkpoint["mu"], checkpoint["sd"]
    t = checkpoint["t_local"] if times is None else times
    T = t.shape[0]
    branch = torch.cat([torch.sin(theta0).view(1, 1), torch.cos(theta0).view(1, 1), ((omega0 - mu) / sd).view(1, 1), Va.view(1, -1), Vb.view(1, -1), Vc.view(1, -1)], dim=-1)

    if model.output_dim == 2:
        # theta and omega are DIRECT network outputs here. compute_theta_omega would
        # build a double-backward graph and take two autograd.grad's whose results are
        # then discarded -- measured 3x of the inference cost (58.0 -> 19.4 ms/sim-s).
        # Kp * (sensor noise) never enters the prediction either way.
        with torch.no_grad():
            out = model(branch, build_trunk_input(t.view(1, T, 1), model.F, model.max_freq))
        return (out[0, :, 0] + theta0 + OMEGA_BASE * t).detach(), out[0, :, 1].detach()

    # single head: omega = dtheta/dt - Kp*Vq, so autograd IS required
    with torch.enable_grad():
        t_query = t.view(1, T, 1).clone().requires_grad_(True)
        out = compute_theta_omega(model, t_query, branch, torch.zeros(1, T, 1), omega_nominal=0.0)
    theta_abs = out["theta"].squeeze() + theta0 + OMEGA_BASE * t     # shift

    # Vq is not measurable at deployment, so recompute it from our own angle estimate
    dtheta_dt = out["omega"].squeeze()          # Vq=0 was passed in, so this is dtheta/dt
    ts = checkpoint["t_local"]
    same = t.shape == ts.shape and torch.equal(t, ts)
    Va_t = Va if same else interp1d(ts, Va, t)
    Vb_t = Vb if same else interp1d(ts, Vb, t)
    Vc_t = Vc if same else interp1d(ts, Vc, t)
    _, Vq = _phys.park_dqTransform(Va_t, Vb_t, Vc_t, theta_abs)
    omega = dtheta_dt - KP * Vq

    return theta_abs.detach(), omega.detach()



def rollout(model, checkpoint, prep, run_idx, n_windows, W):
    if n_windows > W:
        raise ValueError(f"n_windows={n_windows} > W={W}: the rollout would walk " f"into the next run, which has a different grid scenario")
    t_local = checkpoint["t_local"]
    dt = float(t_local[1] - t_local[0])
    t_ext = torch.cat([t_local, t_local[-1:] + dt])
    row0 = run_idx * W
    theta0, omega0 = prep["theta0_abs"][row0], prep["omega0"][row0]
    th_all, om_all = [], []
    for k in range(n_windows):
        th, om = predict_window(model, checkpoint, theta0, omega0, prep["Va"][row0 + k], prep["Vb"][row0 + k], prep["Vc"][row0 + k], t_ext)
        th_all.append(th[:-1]); om_all.append(om[:-1])
        theta0, omega0 = th[-1], om[-1]  # the feedback
    return torch.cat(th_all), torch.cat(om_all)


def truth(prep, run_idx, n_windows, W=windows):
    row0 = run_idx * W
    return (torch.cat([prep["theta_abs"][row0 + k] for k in range(n_windows)]), torch.cat([prep["target_omega"][row0 + k] for k in range(n_windows)]))

def rollout_metrics(model, checkpoint, prep, val_runs, W, n_runs_avg=20):
    """Full-horizon recurrent rollout AND the no-feedback control, in one pass.
    2*W predict_window calls per run -- not 2*sum(1..W)."""
    t_local = checkpoint["t_local"]
    dt = float(t_local[1] - t_local[0])
    t_ext = torch.cat([t_local, t_local[-1:] + dt])
    fed_e, tru_e = [], []
    for r in val_runs[:n_runs_avg]:
        row0 = r * W
        th0, om0 = prep["theta0_abs"][row0], prep["omega0"][row0]
        pf, pt = [], []
        for k in range(W):
            th, om = predict_window(model, checkpoint, th0, om0, prep["Va"][row0 + k],
                                    prep["Vb"][row0 + k], prep["Vc"][row0 + k], t_ext)
            pf.append(th[:-1]); th0, om0 = th[-1], om[-1]          # the feedback
            th2, _ = predict_window(model, checkpoint,
                                    prep["theta0_abs"][row0 + k], prep["omega0"][row0 + k],
                                    prep["Va"][row0 + k], prep["Vb"][row0 + k],
                                    prep["Vc"][row0 + k], t_ext)
            pt.append(th2[:-1])                                     # true IC, no chaining
        truth_th = torch.cat([prep["theta_abs"][row0 + k] for k in range(W)])
        fed_e.append(torch.cat(pf) - truth_th)
        tru_e.append(torch.cat(pt) - truth_th)
    per = lambda es: torch.stack([e.pow(2).mean().sqrt() for e in es])
    rms = lambda es: float(per(es).mean())
    f, t = rms(fed_e), rms(tru_e)
    return {"rollout_full_med": float(per(fed_e).median()),
            "rollout_full_rms": f,
            "rollout_full_max": float(torch.stack([e.abs().max() for e in fed_e]).max()),
            "per_window_rms":   t,
            "compounding":      f / t}

if __name__ == "__main__":
    model, checkpoint = load_checkpoint("pll_deeponet.pth")
    data, meta = Dataset_Creator.load_dataset("pll_dataset.npz")
    prep = prepare(data)
    W = meta["W"]
    tr, va = group_split(prep["run_id"], 0.15, 0)
    val_runs = sorted(set(prep["run_id"][va].tolist()))
    r = val_runs[0]

    #the plot that matters
    for n in [1, 2, 5, 10, 20, 40]:
        thr, _ = rollout(model, checkpoint, prep, r, n, W)
        tht, _ = truth(prep, r, n, W)
        e = thr - tht
        print(f"{n:3d} windows  RMS {torch.sqrt((e**2).mean()):.4e}  max {e.abs().max():.4e} rad")