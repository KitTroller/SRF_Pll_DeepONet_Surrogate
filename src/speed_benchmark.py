import time
import torch
import importlib.util
import os
import sys
import time

import numpy as np
from pathlib import Path

from dataset_generator import Dataset_Creator
from train_pll import load_checkpoint, prepare, OMEGA_BASE
from pll_residual import build_trunk_input, compute_theta_omega

Ignasi_flag = True
from paths import PAPER_REPO, GRAPHS
REPO = str(PAPER_REPO)

def timed(fn, repeats=3):
    fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0)/ repeats

def time_solver(batch):
    torch.set_default_dtype(torch.float64)
    from PLL_Simulator import PLLSimulator
    simulator = PLLSimulator()
    simulator.n_runs = batch

    zeros = torch.zeros(batch, 1)
    Va, Vb, Vc = simulator._grid_phases(zeros, zeros, zeros)
    theta_0, omega_0 = torch.zeros(batch), torch.zeros(batch)
    def run():
        with torch.no_grad():
            simulator.simulate_batch(Va, Vb, Vc, theta_0, omega_0, scheme="trapezoid")
    return timed(run, repeats=1)
def time_surrogate(model, checkpoint, prep, W, batch, lean):
    """W recurrent windows = the same 0.5 s.
       lean=False -> the deployed path, autograd on (what predict_window does today)
       lean=True  -> forward only. Valid because output_dim=2 gives theta and omega
       directly; the two autograd.grad calls are computed and discarded."""
    torch.set_default_dtype(torch.float32)
    mu, sd, t = checkpoint["mu"], checkpoint["sd"], checkpoint["t_local"]
    rows = torch.arange(batch) * W
    Va, Vb, Vc = prep["Va"][rows], prep["Vb"][rows], prep["Vc"][rows]
    theta_init, omega_init = prep["theta0_abs"][rows], prep["omega0"][rows]
    def run():
        theta_0, omega_0 = theta_init.clone(), omega_init.clone()
        for _ in range(W):
            branch = torch.cat([torch.sin(theta_0).unsqueeze(-1), torch.cos(theta_0).unsqueeze(-1), ((omega_0 - mu) / sd).unsqueeze(-1), Va, Vb, Vc], dim=-1)
            tq = t.view(1, -1, 1).expand(batch, -1, 1).clone()
            if lean:
                with torch.no_grad():
                    out = model(branch, build_trunk_input(tq, model.F, model.max_freq))
                    theta, omega = out[..., 0:1], out [..., 1:2]
            else:
                tq.requires_grad_(True)
                o = compute_theta_omega(model, tq, branch, torch.zeros_like(tq), omega_nominal=0.0)
                theta, omega = o["theta"], o["omega"]
            theta_0 = (theta[:, -1, 0] + theta_0 + OMEGA_BASE * t[-1]).detach()
            omega_0 = omega[:, -1, 0].detach()
    return timed(run)

def load_paper_solver(name, sim_step):
    """Import solver_<name>.py under its own module name.

    Their init_sim_saver does `assert sim_step <= sim_plot_step` on a BARE name --
    a module global that exists only when the file runs as __main__. The file
    therefore cannot be imported as shipped; we inject the value afterwards.
    Same class of bug as the two you already list in notes.md -- tell Ignacio."""
    if REPO not in sys.path:
        sys.path.insert(0, REPO)                     # for `from src.… import …`
    spec = importlib.util.spec_from_file_location(
        f"paper_{name}", os.path.join(REPO, f"solver_{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mod.sim_step = sim_step
    return mod


def run_paper(name, sim_time=5.0, sim_step=50e-6, plot_step=500e-6,
              config="params_pv_4events"):
    here = os.getcwd()
    os.chdir(REPO)                                   # their paths are ./sim_setups/...
    try:
        mod = load_paper_solver(name, sim_step)
        sim = mod.EMT_simulator(sim_time, sim_step)
        sim.upload_parameters(plot_step, config)
        if name == "nr":
            sim.create_sim_variables(["x1", "x2", "vq"])   # builds the sympy Jacobian
        if name == "mlp":
            sim.init_neural_net("version1")
        sim.run_algorithm()                          # NOT save_arrays -- see below
        return sim.compute_time / sim_time * 1000    # ms per simulated second
    finally:
        os.chdir(here)


def paper_pll_only(sim_step=50e-6):
    """Their NN alone, no control block around it: 5 in, 2 out, 450 params."""
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    from src.nnet_eval_routine import explicit_inference_f
    net = explicit_inference_f(os.path.join(REPO, "NN_models", "version1"), sim_step)
    x = np.zeros(5, dtype=np.float64)
    per_call = timed(lambda: [net.run(x) for _ in range(1000)]) / 1000
    return per_call * (1.0 / sim_step) * 1000        # ms per simulated second


def ours(batch=1):
    from dataset_generator import Dataset_Creator
    from train_pll import load_checkpoint, prepare
    from speed_benchmark import time_surrogate       # reuse what you already wrote

    data, meta = Dataset_Creator.load_dataset("pll_dataset_W40.npz")
    prep = prepare(data)
    model, ck = load_checkpoint(
        "runs/pll_dataset_W40_n1000_W40_F4_mf503_wp0.3_s0sp0.pth")
    W = meta["W"]
    horizon = W * meta["S"] * meta["dt"]             # 0.5 s of simulated time
    dep  = time_surrogate(model, ck, prep, W, batch, lean=False) / horizon * 1000
    lean = time_surrogate(model, ck, prep, W, batch, lean=True) / horizon * 1000
    return dep, lean

# ===================================================================== accuracy
# The speed numbers cannot settle anything on their own: ours emits one sample per
# 100 us against their 50 us, so "2x faster per simulated second" is just the step
# size. The question that decides it is whether the COARSER STEP COSTS ACCURACY.
#
# Three error sources have to be separated, or the answer is meaningless:
#   discretisation  solver@100us vs a much finer solve  -- paid by ANY method at 100us
#   step halving    solver@50us  vs the same fine solve  -- what their step buys
#   surrogate       DeepONet@100us vs the fine solve     -- what we actually deliver
# If the surrogate error sits at or below the 50us solver error, the coarse step is
# genuinely free. If it sits far above both, the network is the binding constraint
# and the step size is irrelevant -- which is also a result, just a different one.

def build_case(dt_fine, horizon, n_runs, seed=0, their_range=False):
    """ONE grid-voltage realisation on a fine grid. Coarser grids SUBSAMPLE this
    same waveform rather than regenerating it -- `_grid_phases` draws fresh
    torch.rand noise per sample, so a regenerated 50 us run would see a different
    physical signal and the comparison would be meaningless."""
    torch.set_default_dtype(torch.float64)
    from PLL_Simulator import PLLSimulator
    torch.manual_seed(seed)

    n_fine = int(round(horizon / dt_fine))
    sim = PLLSimulator(dt=dt_fine)
    sim.N, sim.n_runs = n_fine, n_runs
    sim.t = (torch.arange(n_fine) * dt_fine).reshape(1, n_fine)

    # same ranges as config/initial_conditions.yml, dependent-flag logic included.
    #
    # their_range=True restricts to what THEIR released network was trained on. It is
    # NOT "near the edge" -- it is the normal locked operating region, and it is sized
    # by measurement, not by guesswork. Their scaler says vq in +/-0.30; sweeping the
    # initial phase error and simulating the FULL trajectory (overshoot included) gives
    # p99|Vq| = 0.278 at eps_half = 0.05*pi (9 deg), 0.343 at 0.07*pi. So 0.05*pi is the
    # widest setting that stays inside their range -- it CANNOT be widened without
    # testing their network on extrapolation.
    # What fills the budget is the phase error, not our sensor noise: at perfect lock
    # p99|Vq| is 0.061 with our noise and 0.031 without it.
    #
    # their_range=True restricts to the INTERSECTION of our envelope and the paper NN's:
    # its scaler says it was trained on x1 in +/-0.05 (= omega +/-15) and vq in +/-0.3,
    # which only holds near lock. Our full range drives vq to +/-1.14, i.e. 3.8x outside
    # anything it ever saw -- comparing there measures extrapolation, not accuracy.
    grid_ang = (torch.rand(n_runs, 1) * 2 - 1) * torch.pi
    freq_off = (torch.rand(n_runs, 1) * 2 - 1) * 0.2
    amp_off  = (torch.rand(n_runs, 1) * 2 - 1) * 0.05
    eps_half = 0.05 * torch.pi if their_range else 0.5 * torch.pi
    om_half  = 12.0 if their_range else 20.0
    theta0   = grid_ang.squeeze(-1) + eps_half * (torch.rand(n_runs) * 2 - 1)
    theta0   = (theta0 + torch.pi) % (2 * torch.pi) - torch.pi
    omega0   = (torch.rand(n_runs) * 2 - 1) * om_half

    Va, Vb, Vc = sim._grid_phases(freq_off, amp_off, grid_ang)
    return dict(sim=sim, Va=Va, Vb=Vb, Vc=Vc, theta0=theta0, omega0=omega0,
                dt_fine=dt_fine, n_fine=n_fine, horizon=horizon)


def solve_at(case, dt, timeit=False):
    """Trapezoidal Newton on a grid of step `dt`, subsampling the SAME waveform.
    Returns theta (n_runs, n_steps) and, optionally, ms of compute per simulated second."""
    torch.set_default_dtype(torch.float64)
    from PLL_Simulator import PLLSimulator
    k = int(round(dt / case["dt_fine"]))
    assert abs(k * case["dt_fine"] - dt) < 1e-15, "dt must be a multiple of dt_fine"
    n = case["n_fine"] // k
    Va, Vb, Vc = case["Va"][:, ::k][:, :n], case["Vb"][:, ::k][:, :n], case["Vc"][:, ::k][:, :n]

    sim = PLLSimulator(dt=dt)
    sim.N, sim.n_runs = n, case["sim"].n_runs
    sim.t = (torch.arange(n) * dt).reshape(1, n)

    def run():
        with torch.no_grad():
            return sim.simulate_batch(Va, Vb, Vc, case["theta0"], case["omega0"],
                                      scheme="trapezoid")
    if timeit:
        t0 = time.perf_counter(); theta = run()[0]
        ms = (time.perf_counter() - t0) / case["horizon"] * 1000
        return theta, ms
    return run()[0], None


def deeponet_at(case, model, ck, W, S, dt, lean=True, timeit=False):
    """W recurrent windows over the same horizon, on the same subsampled waveform.
    Uses pll_infer.predict_window unchanged, so this is the deployed path."""
    from pll_infer import predict_window
    torch.set_default_dtype(torch.float32)
    k = int(round(dt / case["dt_fine"]))
    n = W * S
    Va = case["Va"][:, ::k][:, :n].float()
    Vb = case["Vb"][:, ::k][:, :n].float()
    Vc = case["Vc"][:, ::k][:, :n].float()
    t = ck["t_local"]
    t_ext = torch.cat([t, t[-1:] + float(t[1] - t[0])])

    def run():
        out = []
        for r in range(case["sim"].n_runs):
            th0 = case["theta0"][r].float()
            om0 = case["omega0"][r].float()
            per_run = []
            for w in range(W):
                sl = slice(w * S, (w + 1) * S)
                th, om = predict_window(model, ck, th0, om0,
                                        Va[r, sl], Vb[r, sl], Vc[r, sl], t_ext)
                per_run.append(th[:-1])
                th0, om0 = th[-1], om[-1]          # the handover
            out.append(torch.cat(per_run))
        return torch.stack(out)

    if timeit:
        t0 = time.perf_counter(); theta = run()
        ms = (time.perf_counter() - t0) / case["sim"].n_runs / case["horizon"] * 1000
        return theta, ms
    return run(), None


def paper_nn_at(case, dt, timeit=False):
    """Drive THEIR NN over OUR grid voltage. It is a self-contained PLL stepper:
        (x1_n, theta_n, vq_n, valpha_n+1, vbeta_n+1)  ->  (x1_n+1, theta_n+1)
    exactly the inputs their Newton solve gets, so no network coupling is needed.

    State mapping: their x1 is our omega/Ki (their x2_dot = Kp*vq + Ki*x1 + wN is our
    dtheta/dt = Kp*Vq + omega + omega_0). Their vq = -sin(th)*valpha + cos(th)*vbeta is
    algebraically identical to our Park q-row -- verified by expansion, and it is the
    same D1 sign convention.

    They wrap theta into [0, 2pi] every step (`states[1] %= 2*np.pi` in run_algorithm)
    and their scaler confirms it, so we wrap too and unwrap afterwards to compare."""
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    from src.nnet_eval_routine import explicit_inference_f
    from PLL_Simulator import PhysicsEquations
    torch.set_default_dtype(torch.float64)

    Ki = PhysicsEquations().Ki
    k = int(round(dt / case["dt_fine"]))
    n = case["n_fine"] // k
    Va = case["Va"][:, ::k][:, :n]; Vb = case["Vb"][:, ::k][:, :n]; Vc = case["Vc"][:, ::k][:, :n]
    valpha, vbeta = PhysicsEquations().clarke_alphaBetaTransform(Va, Vb, Vc)
    valpha = valpha.numpy(); vbeta = vbeta.numpy()

    net = explicit_inference_f(os.path.join(REPO, "NN_models", "version1"), dt)
    buf = np.empty(5, dtype=np.float64)
    n_runs = case["sim"].n_runs

    def run():
        out = np.empty((n_runs, n))
        for r in range(n_runs):
            x1 = float(case["omega0"][r]) / Ki
            th = float(case["theta0"][r]) % (2 * np.pi)
            vq = -np.sin(th) * valpha[r, 0] + np.cos(th) * vbeta[r, 0]
            out[r, 0] = th
            for j in range(1, n):
                buf[0], buf[1], buf[2] = x1, th, vq
                buf[3], buf[4] = valpha[r, j], vbeta[r, j]
                y = net.run(buf)
                x1, th = float(y[0]), float(y[1]) % (2 * np.pi)
                vq = -np.sin(th) * valpha[r, j] + np.cos(th) * vbeta[r, j]
                out[r, j] = th
            # ours is unwrapped and absolute; theirs is wrapped into [0, 2pi]. Unwrap
            # restores continuity but NOT the branch: a run starting at theta0 = -3.0
            # begins at 3.28 here and the whole trajectory sits 2pi off the reference.
            # Re-anchor to the true starting angle. (Skipping this gave RMS 4.443 rad,
            # which is exactly sqrt(0.5 * (2pi)^2) -- half the runs offset by one turn.)
            out[r] = np.unwrap(out[r])
            out[r] += float(case["theta0"][r]) - out[r, 0]
        return torch.from_numpy(out)

    if timeit:
        t0 = time.perf_counter(); theta = run()
        ms = (time.perf_counter() - t0) / n_runs / case["horizon"] * 1000
        return theta, ms
    return run(), None


def head_to_head(dt_fine=12.5e-6, horizon=0.5, n_runs=24, their_range=True,
                 ckpt="runs/pll_dataset_n5000_W40_n5000_W40_F4_mf503_wp0.3_s1sp0.pth",
                 dataset="pll_dataset_W40.npz"):
    """Their NN vs ours vs the solver, ALL on the same voltage, same reference.

    Run it with their_range=True. With their_range=False their network is extrapolating
    on vq by ~4x and the numbers say nothing about its accuracy."""
    _, meta = Dataset_Creator.load_dataset(dataset)
    W, S, dt_c = meta["W"], meta["S"], meta["dt"]
    case = build_case(dt_fine, horizon, n_runs, their_range=their_range)
    kc = int(round(dt_c / dt_fine))

    theta_ref, _ = solve_at(case, dt_fine)
    ref = theta_ref[:, ::kc][:, :W * S]

    rows = {}
    th, ms = solve_at(case, dt_c / 2, timeit=True)          # what halving dt buys
    rows["solver @50us"] = (th[:, ::2][:, :W * S] - ref, ms)
    th, ms = solve_at(case, dt_c, timeit=True)
    rows["solver @100us"] = (th[:, :W * S] - ref, ms)

    th, ms = paper_nn_at(case, 50e-6, timeit=True)          # their own step size
    rows["their NN @50us"] = (th[:, ::2][:, :W * S] - ref, ms)
    th, ms = paper_nn_at(case, dt_c, timeit=True)           # forced onto our step
    rows["their NN @100us"] = (th[:, :W * S] - ref, ms)

    torch.set_default_dtype(torch.float32)
    model, ck = load_checkpoint(ckpt)
    th, ms = deeponet_at(case, model, ck, W, S, dt_c, timeit=True)
    rows["ours @100us"] = (th.double() - ref, ms)

    print(f"\nD. head to head, {'INSIDE THEIR TRAINING RANGE' if their_range else 'FULL'} envelope, "
          f"{n_runs} runs x {horizon} s, vs a {dt_fine*1e6:.1f} us reference")
    rms = {}
    for k, (e, ms) in rows.items():
        rms[k] = float(e.pow(2).mean().sqrt())
        print(f"   {k:18s} theta RMS {rms[k]:.3e}   "
              f"max {float(e.abs().max()):.3e}   cost {ms:8.1f} ms/sim-s")

    return rows


def accuracy_benchmark(dt_fine=12.5e-6, horizon=0.5, n_runs=24,
                       dataset="pll_dataset_W40.npz", ckpt=None):
    """`ckpt` may be one path or a LIST. Give it a list.

    A single checkpoint hides the seed variance: the six W=40 mf=503 checkpoints on
    disk span 2.90x on rollout_full_rms, so a one-checkpoint accuracy number carries
    ~3x of uncertainty from that choice alone. Same lesson as n_eval=20 and the
    4-run version of this benchmark -- report a band, never a point.

    Ceiling note: the network is trained to reproduce solver@100us, so it cannot be
    more accurate than solver@100us. Ratio 1.00 is the floor, not a target to beat."""
    if ckpt is None:
        ckpt = ["runs/pll_dataset_W40_n1000_W40_F4_mf503_wp0.3_s0sp0.pth"]
    if isinstance(ckpt, str):
        ckpt = [ckpt]
    import matplotlib.pyplot as plt

    _, meta = Dataset_Creator.load_dataset(dataset)
    W, S, dt_c = meta["W"], meta["S"], meta["dt"]          # 40, 125, 100 us

    case = build_case(dt_fine, horizon, n_runs)
    kc = int(round(dt_c / dt_fine))                        # fine -> 100 us decimation

    theta_ref, _  = solve_at(case, dt_fine)                # the truth
    theta_50, m50 = solve_at(case, dt_c / 2, timeit=True)  # their step
    theta_100, m100 = solve_at(case, dt_c,   timeit=True)  # our step

    ref = theta_ref[:, ::kc][:, :W * S]                    # compare on the 100 us grid
    err = {"solver @50us":  (theta_50[:, ::2][:, :W * S] - ref),
           "solver @100us": (theta_100[:, :W * S]        - ref)}
    cost = {"solver @50us": m50, "solver @100us": m100}

    for path in ckpt:
        # load_checkpoint rebuilds the MLP with the CURRENT default dtype, and
        # solve_at just left it at float64 -- load_state_dict then copies the saved
        # float32 weights into float64 layers and inference dies on a dtype mismatch.
        torch.set_default_dtype(torch.float32)
        model, ck = load_checkpoint(path)
        theta_nn, mnn = deeponet_at(case, model, ck, W, S, dt_c, timeit=True)
        # must be UNIQUE per checkpoint: splitting on "_W40_" collapsed the n=1000 and
        # n=5000 seed-0 runs onto the same key and silently dropped one of them.
        tag = "DeepONet " + Path(path).stem.replace("pll_dataset_", "")
        err[tag] = theta_nn.double() - ref
        cost[tag] = mnn

    print(f"\nC. accuracy vs a {dt_fine*1e6:.1f} us reference, "
          f"{n_runs} runs x {horizon} s   [theta RMS, rad]")
    rms = {}
    for k, e in err.items():
        rms[k] = float(e.pow(2).mean().sqrt())
        print(f"   {k:44s} RMS {rms[k]:.3e}   max {float(e.abs().max()):.3e}"
              f"   cost {cost[k]:8.1f} ms/sim-s")

    nn_keys = [k for k in rms if k.startswith("DeepONet")]
    if len(nn_keys) > 1:
        lo, hi = min(rms[k] for k in nn_keys), max(rms[k] for k in nn_keys)
        base = rms["solver @100us"]
        print(f"\n   across {len(nn_keys)} checkpoints: {lo:.3e} .. {hi:.3e}  "
              f"({hi/lo:.2f}x spread) = {lo/base:.2f}x .. {hi/base:.2f}x the solver "
              f"at the SAME step (1.00x is the ceiling -- it learned from that solver)")

    t_axis = np.arange(W * S) * dt_c
    GRAPHS.mkdir(exist_ok=True)

    # SINGLE panel since 2026-08-20. The cost-vs-accuracy scatter that used to sit on the
    # right was a duplicate of graphs/12, which draws the same plot AND includes the
    # paper's own network -- strictly the better version. What is unique here is the TIME
    # axis: WHERE in the 0.5 s the error lives, which no other figure shows.
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for k, e in err.items():
        ax.semilogy(t_axis, e.abs().mean(dim=0).numpy(), lw=1.0, label=k)
    ax.set_xlabel("time [s]"); ax.set_ylabel(r"$|\theta$ error$|$, mean over runs [rad]")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax.set_title(f"Where the error lives in time — vs a {dt_fine*1e6:.1f} $\\mu$s "
                 f"trapezoidal reference\n{n_runs} runs x {horizon} s. "
                 f"Cost vs accuracy is graphs/12.", fontsize=10)
    fig.tight_layout(); fig.savefig(GRAPHS / "09_accuracy_vs_step.png", dpi=160)
    print(f"-> {GRAPHS / '09_accuracy_vs_step.png'}")
    return rms, cost


if __name__ == "__main__":
    if not Ignasi_flag:
        DATASET = "pll_dataset_n5000_W40.npz"
        CKPT    = "runs/pll_dataset_n5000_W40_n5000_W40_F4_mf503_wp0.3_s0sp0.pth"

        data, meta = Dataset_Creator.load_dataset(DATASET)
        prep = prepare(data)
        model, ck = load_checkpoint(CKPT)
        W = meta["W"]

        print(f"{'batch':>6s} {'solver':>10s} {'deployed':>10s} {'lean':>10s} "
            f"{'x(depl)':>8s} {'x(lean)':>8s}   seconds per 0.5 s of simulated time")
        for batch in (1, 10, 100, 1000, 3000, 5000):
            s  = time_solver(batch)
            d  = time_surrogate(model, ck, prep, W, batch, lean=False)
            ln = time_surrogate(model, ck, prep, W, batch, lean=True)
            print(f"{batch:6d} {s:10.4f} {d:10.4f} {ln:10.4f} {s/d:8.1f} {s/ln:8.1f}")
    else:
        SIM_TIME = 5.0        # the paper's own setting for params_pv_4events

        print("A. whole control block (their metric)  [ms compute / simulated second]")
        for name in ("nr", "del", "mlp"):
            print(f"   {name:34s} {run_paper(name, SIM_TIME):10.1f}")

        print("\nB. PLL component only                  [ms compute / simulated second]")
        print(f"   {'their NN  450 par, 50us step':34s} {paper_pll_only():10.1f}")
        dep, lean = ours()
        print(f"   {'ours deployed  45.7k, 12.5ms win':34s} {dep:10.1f}")
        print(f"   {'ours lean      45.7k, 12.5ms win':34s} {lean:10.1f}")

        accuracy_benchmark()