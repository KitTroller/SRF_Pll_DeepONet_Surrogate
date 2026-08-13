import torch
from omegaconf import OmegaConf
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent / "config"
pll_constants = OmegaConf.load(CONFIG_DIR / "PLL_Constants.yml")
model_config = OmegaConf.load(CONFIG_DIR / "DeepONet_models.yml")


def build_trunk_input(t, F=model_config.num_fourier_feats, max_F_freq=model_config.max_fourier_feat_frequency ):
    feats =[t]
    if F > 0:
        for k in range(1, F + 1):
            w = max_F_freq * k / F
            feats += [torch.sin(w * t), torch.cos(w * t)]
        
    return torch.cat(feats, dim=-1)
    
def compute_theta_omega(model, t_query, branch, Vq, omega_nominal=None):
    Ki = pll_constants.Pll.Ki
    Kp = pll_constants.Pll.Kp
    if omega_nominal is None:
        omega_0 = 2 * torch.pi * pll_constants.Pll.f_0
    else:
        omega_0 = omega_nominal
    trunk = build_trunk_input(t_query)
    theta = model.forward(branch, trunk)
    dtheta_dt = torch.autograd.grad(theta, t_query, grad_outputs=torch.ones_like(theta), create_graph=True)[0]
    omega = dtheta_dt - Kp * Vq - omega_0
    domega_dt = torch.autograd.grad(omega, t_query, grad_outputs=torch.ones_like(omega),create_graph=True)[0]
    residual = domega_dt - Ki * Vq
    
    return {"theta": theta, "omega": omega, "residual": residual}