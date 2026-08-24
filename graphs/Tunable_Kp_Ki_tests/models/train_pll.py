import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from pytorch_optimizer import SOAP
from pathlib import Path
import time
import json
import numpy as np

from paths import sweeps as _sweeps
from dataset_generator import Dataset_Creator
from pll_operator import Unstacked_DeepONet, Single_PINN
from pll_residual import compute_theta_omega

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"   # src/ -> root
torch.set_default_dtype(torch.float32)
pll_constants = OmegaConf.load(CONFIG_DIR / "PLL_Constants.yml")
OMEGA_BASE = 2 * torch.pi * pll_constants.Pll.f_0
KP, KI = pll_constants.Pll.Kp, pll_constants.Pll.Ki
DEVICE = ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

class PLLDataset(Dataset):
    """
    data = {                                                                #check dataset_generator
        "theta_pll": theta_pll,
        "omega_pll": omega_pll,
        "Vd":        Vd,
        "Vq":        Vq,
        "Valpha":    Valpha,
        "Vbeta":     Vbeta,
        "Va":        Va,
        "Vb":        Vb,
        "Vc":        Vc,
        "t_local"    = self.t_local                       # (S,)   shared
        "theta0"     = windowed["theta_pll"][:, 0]        # (n_samples,)
        "omega0"     = windowed["omega_pll"][:, 0]
        "run_id"     = torch.arange(self.n_runs).repeat_interleave(W)
        "segment_id" = torch.arange(W).repeat(self.n_runs)
        "lhs_samples" = init_conditions                   # (n_runs, 5)
    }
    meta = {                                                            #from build_meta in dataset_generator
        "dt": float(sim.dt), "N": int(sim.N), "W": int(self.W),
        "S": int(sim.N // self.W), "n_runs": int(self.n_runs),
        "Kp": float(ph.Kp), "Ki": float(ph.Ki), "omega_0": float(ph.omega_0),
        "v_nominal": float(ph.v_nominal),
        "noise_amplitude": float(ph.noise_amplitude),
        "columns": ["initial_grid_angle", "frequency_offset", "amplitude_offset", "theta_pll", "omega_pll"],
        "ranges": OmegaConf.to_container(self.init_cond.ranges, resolve=True),
        "forcing_channels": ["Va", "Vb", "Vc"],
    }  
    """
    KEYS = ["sin_theta_0", "cos_theta_0", "omega0", "Va", "Vb", "Vc", "Vq", "target_theta", "target_omega"]
    def __init__(self, prep, mask):
        self.t = {k: prep[k][mask] for k in self.KEYS}
        
    def __len__(self):
        return self.t["omega0"].shape[0]
    
    def __getitem__(self, i):
        return tuple(self.t[k][i] for k in self.KEYS)

def prepare(data, deviation=True):
    t = data["t_local"]
    out = {}
    if deviation:
        
        print(f"theta 0 shape: {data["theta0"].shape} theta_0 unsqueezed shape: {data["theta0"].unsqueeze(-1).shape}")
        print(f"t shape: {t.shape} t unsqueezed shape: {t.unsqueeze(0).shape}")
        
        ramp = data["theta0"].unsqueeze(-1) + OMEGA_BASE * t.unsqueeze(0)
        out["target_theta"] = (data["theta_pll"] - ramp).float()
    else:
        out["target_theta"] = data["theta_pll"].float()
    out["target_omega"] = data["omega_pll"].float()
    out["sin_theta_0"] = torch.sin(data["theta0"]).float()
    out["cos_theta_0"] = torch.cos(data["theta0"]).float()
    for k in ["omega0", "Va", "Vb", "Vc", "Vq"]:
        out[k] = data[k].float()
    out["t_local"] = t.float()
    out["run_id"] = data["run_id"]
    if "kp" in data:                       # per-run controller gains, when sampled
        out["kp"] = data["kp"].float()
        out["ki"] = data["ki"].float()
    out["theta0_abs"] = data["theta0"].float()
    out["theta_abs"]  = data["theta_pll"].float()
    return out 

def group_split(run_id, val_frac=0.15, seed=0):
    runs = torch.unique(run_id)
    g = torch.Generator().manual_seed(seed)
    permutations = runs[torch.randperm(len(runs), generator=g)]
    n_val = max(1, int(round(len(runs) * val_frac)))
    validation_runs = set(permutations[:n_val].tolist())
    val_mask = torch.tensor([int(r) in validation_runs for r in run_id])
    return ~val_mask, val_mask 

def build_branch(prep, mu, sd, gstat=None):
    cols = [prep["sin_theta_0"].unsqueeze(-1), prep["cos_theta_0"].unsqueeze(-1), ((prep["omega0"] - mu) / sd).unsqueeze(-1), prep["Va"], prep["Vb"], prep["Vc"]]
    if gstat is not None:      # APPENDED, so the Va/Vb/Vc offsets stay at 3, 3+S, 3+2S
        (kmu, ksd), (imu, isd) = gstat
        cols += [((prep["kp"] - kmu) / ksd).unsqueeze(-1), ((prep["ki"] - imu) / isd).unsqueeze(-1)]
    return torch.cat(cols, dim=-1)      # (n, 3 + 3S [+ 2])

def batches(tensors, n, batch_size, shuffle, device, drop_last):
    idx = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
    stop = n - (n % batch_size) if drop_last else n
    for i in range(0, stop, batch_size):
        b = idx[i:i + batch_size]
        yield tuple(t[b] for t in tensors)
        
def assemble_batch(batch, t_local, mu, sd):  # DO NOT READ UNUSED should be deleted kept for emotional purposes
    sin_theta_0, cos_theta_0, omega0, Va, Vb, Vc, Vq, target_theta, target_omega = batch  # is a dataloader instance of 64 elements (B, n_runs, 500)
    B = omega0.shape[0]
    branch = torch.cat([sin_theta_0.unsqueeze(-1), cos_theta_0.unsqueeze(-1), ((omega0 - mu) / sd).unsqueeze(-1), Va, Vb, Vc], dim=-1)
    t_query = t_local.view(1, -1, 1).expand(B, -1, 1).clone().requires_grad_(True)
    # print(f"Shapes are: branch: {branch.shape}, t_query: {t_query.shape}, Vq: {Vq.unsqueeze(-1).shape}, target_theta: {target_theta.unsqueeze(-1).shape}, target_omega: {target_omega.unsqueeze(-1).shape}")
    return branch, t_query, Vq.unsqueeze(-1), target_theta.unsqueeze(-1), target_omega.unsqueeze(-1)

def run_epoch(model, tensors, n, t_local, w_omega, w_phys, s1, s2, batch_size, optimizer=None, device="cpu", residual="eq4"):
    is_train = optimizer is not None
    model.train(is_train)
    tot = {"theta": 0.0, "omega": 0.0, "r1": 0.0, "r2": 0.0, "phys": 0.0, "total": 0.0}    
    nb = 0
    for branch, Vq, tth, tom, kp, ki in batches(tensors, n, batch_size, is_train, device, is_train):
        B = branch.shape[0]
        t_query = t_local.view(1, -1, 1).expand(B, -1, 1).clone().requires_grad_(True)
        # (B,1,1) so they broadcast against Vq (B,T,1). When the dataset has no gains
        # these columns are filled with the YAML scalars, so the maths is identical.
        out  = compute_theta_omega(model, t_query, branch, Vq.unsqueeze(-1), omega_nominal=0.0,
                                   residual=residual, Kp=kp.view(-1, 1, 1), Ki=ki.view(-1, 1, 1))
        l_th = nn.functional.mse_loss(out["theta"], tth.unsqueeze(-1))
        l_om = nn.functional.mse_loss(out["omega"], tom.unsqueeze(-1))
        # each residual divided by the RMS of the terms it is BUILT from, so both
        # are dimensionless. 1.0 means "as big as the equation's own terms".
        r1 = (out["res_theta"] / s1).pow(2).mean()
        r2 = (out["res_omega"] / s2).pow(2).mean()
        l_ph = r1 + r2
        loss = l_th + w_omega * l_om + w_phys * l_ph
        if is_train:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        tot["theta"] += l_th.item(); tot["omega"] += l_om.item()
        tot["r1"] += r1.item();      tot["r2"] += r2.item()
        tot["phys"] += l_ph.item();  tot["total"] += loss.item(); nb += 1
    return {k: v / nb for k, v in tot.items()}

def save_checkpoint(path, model, meta, mu, sd, w_omega, s1, s2, t_local, history, best_epoch, w_phys=None, gstat=None):
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "cfg": model.config(),
                "mu": mu.cpu(), "sd": sd.cpu(), "w_omega": w_omega, "gstat": gstat,
                "s1": s1, "s2": s2,
                "t_local": t_local.cpu(), "deviation": True,
                "omega_base": OMEGA_BASE, "data_meta": meta,
                "history": history, "best_epoch": best_epoch, "w_phys": w_phys}, path)
    
def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cls = {"Unstacked_DeepONet": Unstacked_DeepONet, "Single_PINN": Single_PINN}[ck["cfg"]["arch"]]
    model = cls(cfg=ck["cfg"])          # rebuilt from the checkpoint
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    ck["t_local"] = ck["t_local"].to(device)
    return model, ck



ARCHS = {"deeponet": Unstacked_DeepONet, "pinn": Single_PINN}


def main(dataset="pll_dataset.npz", epochs=200, lr=3e-3, w_phys=0.0, batch_size=512, val_frac=0.15, patience=20, seed=0, split_seed=0, F=None, out=None, device=DEVICE, n_eval_runs=20, results_dir="sweeps", runs_dir="runs", max_freq=None, hidden_dim=None, arch="deeponet", residual="eq4"):
    torch.manual_seed(seed)
    data, meta = Dataset_Creator.load_dataset(dataset)
    prep = prepare(data, deviation=True)

    tr, va = group_split(prep["run_id"], val_frac, split_seed)
    assert not (set(prep["run_id"][tr].tolist()) & set(prep["run_id"][va].tolist()))
    print(f"device {device} | {tr.sum().item()} train / {va.sum().item()} val rows")

    mu = prep["omega0"][tr].mean()
    sd = prep["omega0"][tr].std()
    w_omega = (1.0 / prep["target_omega"][tr].var()).item()
    # residual scales = RMS of the terms each equation is made of
    s1 = (prep["target_omega"][tr] + KP * prep["Vq"][tr]).pow(2).mean().sqrt().item()   # rad/s
    s2 = (KI * prep["Vq"][tr]).pow(2).mean().sqrt().item()                              # rad/s^2
    print(f"mu={mu:.4f} sd={sd:.4f} w_omega={w_omega:.4e} | s1={s1:.3f} rad/s  s2={s2:.3f} rad/s^2")

    has_gains = "kp" in prep
    gstat = None
    if has_gains:
        kmu, ksd = prep["kp"][tr].mean(), prep["kp"][tr].std()
        imu, isd = prep["ki"][tr].mean(), prep["ki"][tr].std()
        gstat = ((kmu, ksd), (imu, isd))
        print(f"per-run gains ON: Kp {prep['kp'].min():.1f}-{prep['kp'].max():.1f} "
              f"(mu {kmu:.1f} sd {ksd:.1f}) | Ki {prep['ki'].min():.0f}-{prep['ki'].max():.0f} "
              f"(mu {imu:.0f} sd {isd:.0f})")
    branch = build_branch(prep, mu, sd, gstat)
    # per-row Kp/Ki for the residual: from the data when sampled, else the YAML scalars
    kp_col = prep["kp"] if has_gains else torch.full_like(prep["omega0"], float(KP))
    ki_col = prep["ki"] if has_gains else torch.full_like(prep["omega0"], float(KI))
    pack = lambda m: tuple(t[m].to(device) for t in (branch, prep["Vq"], prep["target_theta"], prep["target_omega"], kp_col, ki_col))
    tr_t, va_t = pack(tr), pack(va)
    n_tr, n_va = int(tr.sum()), int(va.sum())
    t_local = prep["t_local"].to(device)

    ov = {"S_win": meta["S"], "n_extra": 2 if has_gains else 0}
    if F is not None:
        ov["F"] = F
    if max_freq is not None: ov["max_freq"] = max_freq
    if hidden_dim is not None: ov["hidden_dim"] = hidden_dim
    model = ARCHS[arch](ov=ov).to(device)

    
    tag = (f"{Path(dataset).stem}_n{meta['n_runs']}_W{meta['W']}_F{model.F}" f"_mf{model.max_freq:g}_wp{w_phys:g}_s{seed}sp{split_seed}"
           + (f"_h{hidden_dim}" if hidden_dim is not None else "")
           + ("" if arch == "deeponet" else f"_{arch}")
           + ("" if residual == "eq4" else f"_{residual}")
           + ("_g" if has_gains else ""))
    results_dir = _sweeps(results_dir)
    Path(runs_dir).mkdir(exist_ok=True); Path(results_dir).mkdir(parents=True, exist_ok=True)
    out = out or f"{runs_dir}/{tag}.pth"
    shape = (f"branch_in={model.branch_sizes[0]} trunk_in={model.trunk_sizes[0]}"
             if hasattr(model, "branch_sizes") else f"pinn_in={model.pinn_sizes[0]}")
    print(f"[{tag}] arch={arch} residual={residual} output_dim={model.output_dim} {shape} "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    t_start = time.perf_counter()

    opt = SOAP(model.parameters(), lr=lr, betas=(.95, .95), weight_decay=.01, precondition_frequency=10)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(3, patience // 3))

    best, best_state, best_ep, bad = float("inf"), None, -1, 0
    history = {"train": [], "val": []}
    status = "ok"
    for ep in range(1, epochs + 1):
        trm = run_epoch(model, tr_t, n_tr, t_local, w_omega, w_phys, s1, s2, batch_size, opt, device, residual)
        vam = run_epoch(model, va_t, n_va, t_local, w_omega, w_phys, s1, s2, batch_size, None, device, residual)
        if (not np.isfinite(vam["total"])) or (ep > 3 and vam["total"] > 50 * best):
            status = "diverged"
            print(f"[{tag}] DIVERGED at epoch {ep}: val total {vam['total']:.3e} "
                  f"vs best {best:.3e} -- aborting")
            break
        history["train"].append(trm); history["val"].append(vam)
        
        sched.step(vam["total"])
        improved = vam["total"] < best - 1e-9
        if improved:
            best, best_ep, bad = vam["total"], ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"[{ep:4d}] train th {trm['theta']:.3e} om {trm['omega']:.3e} "
              f"r1 {trm['r1']:.3e} r2 {trm['r2']:.3e} | "
              f"val th {vam['theta']:.3e} om {vam['omega']:.3e} "
              f"r1 {vam['r1']:.3e} r2 {vam['r2']:.3e}{'  *' if improved else ''}")
        if bad >= patience:
            print(f"early stop at {ep} (best {best_ep})"); break

    seconds = round(time.perf_counter() - t_start, 1)
    rec = {"tag": tag, "status": status, "dataset": dataset,
           "W": meta["W"], "S": meta["S"], "dt": meta["dt"],
           "window_s": meta["S"] * meta["dt"],
           "F": model.F, "max_freq": model.max_freq, "w_phys": w_phys,
           "seed": seed, "split_seed": split_seed, "lr": lr,
           "arch": arch, "residual": residual, "gains": bool(has_gains),
           "params": sum(p.numel() for p in model.parameters()),
           "batch_size": batch_size, "device": str(device),
           "n_eval_runs": n_eval_runs,
           "epochs_run": len(history["val"]), "seconds": seconds}
    if status == "ok" and best_state is not None:
        model.load_state_dict(best_state)
        save_checkpoint(out, model, meta, mu, sd, w_omega, s1, s2,
                        t_local, history, best_ep, w_phys,
                        None if gstat is None else tuple((float(a), float(b)) for a, b in gstat))
        h = history["val"][best_ep - 1]
        rec.update({"best_epoch": best_ep, "ckpt": out,
                    "val_th": h["theta"], "val_om": h["omega"],
                    "r1": h["r1"], "r2": h["r2"],
                    "train_th": history["train"][best_ep - 1]["theta"]})
        from pll_infer import rollout_metrics          # deferred: circular import
        ev_model, ck = load_checkpoint(out)
        val_runs = sorted(set(prep["run_id"][va].tolist()))
        rec.update(rollout_metrics(ev_model, ck, prep, val_runs, meta["W"], n_eval_runs))
        print(f"[{tag}] rollout_full_rms {rec['rollout_full_rms']:.4e}  "
              f"compounding {rec['compounding']:.2f}x  ({seconds:.0f}s)")
    else:
        rec["status"] = "diverged"                    # also covers best_state=None
        print(f"[{tag}] DIVERGED -- no checkpoint written ({seconds:.0f}s)")

    p = Path(results_dir) / f"{tag}.json"
    tmp = p.with_suffix(".tmp"); tmp.write_text(json.dumps(rec, indent=2)); tmp.replace(p)
    return rec

if __name__ == "__main__":
    main(epochs=200, w_phys=0.0)
