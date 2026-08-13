import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from pytorch_optimizer import SOAP
from pathlib import Path

from dataset_generator import Dataset_Creator
from pll_operator import Unstacked_DeepONet, Single_PINN
from pll_residual import compute_theta_omega

CONFIG_DIR = Path(__file__).resolve().parent / "config"
torch.set_default_dtype(torch.float32)
pll_constants = OmegaConf.load(CONFIG_DIR / "PLL_Constants.yml")
OMEGA_BASE = 2 * torch.pi * pll_constants.Pll.f_0

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

def assemble_batch(batch, t_local, mu, sd):
    sin_theta_0, cos_theta_0, omega0, Va, Vb, Vc, Vq, target_theta, target_omega = batch  # is a dataloader instance of 64 elements (B, n_runs, 500)
    B = omega0.shape[0]
    branch = torch.cat([sin_theta_0.unsqueeze(-1), cos_theta_0.unsqueeze(-1), ((omega0 - mu) / sd).unsqueeze(-1), Va, Vb, Vc], dim=-1)
    t_query = t_local.view(1, -1, 1).expand(B, -1, 1).clone().requires_grad_(True)
    # print(f"Shapes are: branch: {branch.shape}, t_query: {t_query.shape}, Vq: {Vq.unsqueeze(-1).shape}, target_theta: {target_theta.unsqueeze(-1).shape}, target_omega: {target_omega.unsqueeze(-1).shape}")
    return branch, t_query, Vq.unsqueeze(-1), target_theta.unsqueeze(-1), target_omega.unsqueeze(-1)

def run_epoch(model, loader, t_local, mu, sd, w_omega, w_phys, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    tot = {"theta": 0.0, "omega": 0.0, "phys": 0.0, "total": 0.0}
    n = 0
    for batch in loader:
        branch, t_query, Vq, tth, tom = assemble_batch(batch, t_local, mu, sd)  # target theta, target omega
        out = compute_theta_omega(model, t_query, branch, Vq, omega_nominal=0.0)
        l_th = nn.functional.mse_loss(out["theta"], tth)
        l_om = nn.functional.mse_loss(out["omega"], tom)
        l_ph = out["residual"].pow(2).mean()
        loss = l_th + w_omega * l_om + w_phys * l_ph
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tot["theta"] += l_th.item(); tot["omega"] += l_om.item()
        tot["phys"] += l_ph.item();  tot["total"] += loss.item(); n += 1
    return {k: v / n for k, v in tot.items()}

def save_checkpoint(path, model, meta, mu, sd, w_omega, t_local, history, best_epoch):
    torch.save({"state_dict": model.state_dict(),
                "arch": type(model).__name__,
                "mu": mu, "sd": sd, "w_omega": w_omega,
                "t_local": t_local, "deviation": True,
                "omega_base": OMEGA_BASE, "data_meta": meta,
                "history": history, "best_epoch": best_epoch}, path)
    
def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = {"Unstacked_DeepONet": Unstacked_DeepONet,  "Single_PINN": Single_PINN}[ck["arch"]]()
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck


def main(dataset="pll_dataset.npz", epochs=200, lr=1e-3, w_phys=0.0, batch_size=64, val_frac=0.15, patience=20, seed=0, out="pll_deeponet.pth"):
    torch.manual_seed(seed)
    data, meta = Dataset_Creator.load_dataset(dataset)
    prep = prepare(data, deviation=True)

    tr, va = group_split(prep["run_id"], val_frac, seed)  # training and validation data split
    assert not (set(prep["run_id"][tr].tolist()) & set(prep["run_id"][va].tolist()))
    print(f"{tr.sum().item()} train / {va.sum().item()} val rows")

    # normalisation and loss scaling from the TRAINING split only
    mu = prep["omega0"][tr].mean()
    sd = prep["omega0"][tr].std()
    w_omega = 1.0 / prep["target_omega"][tr].var()      # omega term is 126x larger
    print(f"omega0 norm mu={mu:.4f} sd={sd:.4f}   w_omega={w_omega:.4e}")

    t_local = prep["t_local"]
    training_loader = DataLoader(PLLDataset(prep, tr), batch_size=batch_size, shuffle=True, drop_last=True)
    validation_loader = DataLoader(PLLDataset(prep, va), batch_size=batch_size)

    model = Unstacked_DeepONet()
    #opt = torch.optim.Adam(model.parameters(), lr=lr)
    opt = SOAP(model.parameters(), lr=3e-3, betas=(.95, .95), weight_decay=.01, precondition_frequency=10)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(3, patience // 3))

    best, best_state, best_ep, bad = float("inf"), None, -1, 0
    history = {"train": [], "val": []}
    for ep in range(1, epochs + 1):
        trm = run_epoch(model, training_loader, t_local, mu, sd, w_omega, w_phys, opt)  # training model  
        vam = run_epoch(model, validation_loader, t_local, mu, sd, w_omega, w_phys, None) # validation model
        history["train"].append(trm); history["val"].append(vam)
        sched.step(vam["total"])
        improved = vam["total"] < best - 1e-7
        if improved:
            best, best_ep, bad = vam["total"], ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"[{ep:4d}] train th {trm['theta']:.3e} om {trm['omega']:.3e} "f"ph {trm['phys']:.3e} | val th {vam['theta']:.3e} "f"om {vam['omega']:.3e}{'  *' if improved else ''}")
        if bad >= patience:
            print(f"early stop at {ep} (best {best_ep})")
            break

    model.load_state_dict(best_state)
    save_checkpoint(out, model, meta, mu, sd, w_omega, t_local, history, best_ep)
    print(f"saved {out} (best epoch {best_ep})")


if __name__ == "__main__":
    #c1_overfit()
    main(epochs=200, w_phys=5e-8) 