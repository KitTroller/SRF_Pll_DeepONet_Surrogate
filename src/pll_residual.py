import torch
from omegaconf import OmegaConf
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"   # src/ -> root
pll_constants = OmegaConf.load(CONFIG_DIR / "PLL_Constants.yml")
model_config = OmegaConf.load(CONFIG_DIR / "DeepONet_models.yml")


def build_trunk_input(t, F, max_F_freq):
    feats =[t]
    if F > 0:
        for k in range(1, F + 1):
            w = max_F_freq * k / F
            feats += [torch.sin(w * t), torch.cos(w * t)]
        
    return torch.cat(feats, dim=-1)
    
def compute_theta_omega(model, t_query, branch, Vq, omega_nominal=None):
    """physics:  dtheta/dt = omega_0 + omega + Kp*Vq     (eq 1)
                 domega/dt = Ki*Vq                       (eq 2)
    Returns theta, omega and one residual per equation."""
    
    Ki = pll_constants.Pll.Ki
    Kp = pll_constants.Pll.Kp
    if omega_nominal is None:
        omega_0 = 2 * torch.pi * pll_constants.Pll.f_0
    else:
        omega_0 = omega_nominal
    trunk = build_trunk_input(t_query, model.F, model.max_freq)
    out = model.forward(branch, trunk)
    if model.output_dim ==1:
        theta = out
        dtheta_dt = torch.autograd.grad(theta, t_query, grad_outputs=torch.ones_like(theta), create_graph=True)[0]
        omega = dtheta_dt - Kp * Vq - omega_0
        domega_dt = torch.autograd.grad(omega, t_query, grad_outputs=torch.ones_like(omega),create_graph=True)[0]
        residual = domega_dt - Ki * Vq
        return {"theta": theta, "omega": omega, "res_theta": torch.zeros_like(omega), "res_omega": residual}

    theta = out[..., 0:1]
    omega = out[..., 1:2]
    dtheta_dt = torch.autograd.grad(theta, t_query, torch.ones_like(theta), create_graph=True)[0]
    domega_dt = torch.autograd.grad(omega, t_query, torch.ones_like(omega), create_graph=True)[0]
    return {"theta": theta, "omega": omega, "res_theta": dtheta_dt - omega - Kp * Vq - omega_0, "res_omega": domega_dt - Ki * Vq}                      
        