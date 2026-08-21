import torch
from omegaconf import OmegaConf
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"   # src/ -> root
pll_constants = OmegaConf.load(CONFIG_DIR / "PLL_Constants.yml")
model_config = OmegaConf.load(CONFIG_DIR / "DeepONet_models.yml")


OMEGA_BASE = 2 * torch.pi * pll_constants.Pll.f_0


def _park_q(Va, Vb, Vc, theta):
    """Same q-row as PhysicsEquations.park_dqTransform, minus-sign convention (D1)."""
    return -(2 / 3) * (Va * torch.sin(theta)
                       + Vb * torch.sin(theta - (2 / 3) * torch.pi)
                       + Vc * torch.sin(theta + (2 / 3) * torch.pi))


def vq_from_prediction(branch, theta_dev, t_query):
    """eq-6: recompute Vq from our own angle instead of reading it from the dataset.

    eq-4 (the default) hands the residual the stored `Vq`, which is a constant w.r.t.
    the network output. That is what makes the physics loss exactly gauge invariant
    under theta -> theta + a*t + b, omega -> omega + a: `dtheta/dt - omega` and
    `domega/dt` are both unchanged and Vq does not move, so the loss cannot select
    which ODE solution -- it is pure derivative (Sobolev) supervision.

    eq-6 breaks that on purpose. Vq becomes a function of theta, the invariance is
    gone, and the physics term starts carrying information about the solution itself.
    Whether that helps is the open question; it is not obviously good, because it also
    lets the residual be satisfied by a self-consistent wrong angle.

    branch = [sin th0, cos th0, (om0-mu)/sd, Va(S), Vb(S), Vc(S)], so everything needed
    is already there. theta_dev is the deviation form (D5), hence the ramp is added back.
    """
    S = (branch.shape[-1] - 3) // 3
    T = t_query.shape[1]
    if T != S:
        raise ValueError(f"eq-6 needs one query point per sensor: T={T} but S={S}. "
                         "The stored Va/Vb/Vc are only defined at the sensor times.")
    theta0 = torch.atan2(branch[:, 0:1], branch[:, 1:2]).unsqueeze(-1)      # (B,1,1)
    Va = branch[:, 3:3 + S].unsqueeze(-1)
    Vb = branch[:, 3 + S:3 + 2 * S].unsqueeze(-1)
    Vc = branch[:, 3 + 2 * S:3 + 3 * S].unsqueeze(-1)
    return _park_q(Va, Vb, Vc, theta_dev + theta0 + OMEGA_BASE * t_query)


def build_trunk_input(t, F, max_F_freq):
    feats =[t]
    if F > 0:
        for k in range(1, F + 1):
            w = max_F_freq * k / F
            feats += [torch.sin(w * t), torch.cos(w * t)]
        
    return torch.cat(feats, dim=-1)
    
def compute_theta_omega(model, t_query, branch, Vq, omega_nominal=None, residual="eq4", Kp=None, Ki=None):
    """physics:  dtheta/dt = omega_0 + omega + Kp*Vq     (eq 1)
                 domega/dt = Ki*Vq                       (eq 2)
    Returns theta, omega and one residual per equation."""
    
    # Kp/Ki may be per-sample tensors of shape (B,1,1) when the dataset carries per-run
    # controller gains -- they broadcast against Vq (B,T,1). Scalars from the YAML
    # otherwise, so every existing call site is unchanged.
    Ki = pll_constants.Pll.Ki if Ki is None else Ki
    Kp = pll_constants.Pll.Kp if Kp is None else Kp
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
        res = domega_dt - Ki * Vq          # NOT `residual`: that name is now the eq4/eq6 flag
        return {"theta": theta, "omega": omega, "res_theta": torch.zeros_like(omega), "res_omega": res}

    theta = out[..., 0:1]
    omega = out[..., 1:2]
    dtheta_dt = torch.autograd.grad(theta, t_query, torch.ones_like(theta), create_graph=True)[0]
    domega_dt = torch.autograd.grad(omega, t_query, torch.ones_like(omega), create_graph=True)[0]
    if residual == "eq6":                      # Vq from our own angle; see the docstring
        if getattr(model, "n_extra", 0):
            raise NotImplementedError(
                "eq6 parses the branch positionally and has not been updated for the "
                "appended gain columns. eq4 is the better formulation anyway (F53: eq6 is "
                "never better and up to 8x worse), so this combination is not worth wiring.")
        Vq = vq_from_prediction(branch, theta, t_query)
    elif residual != "eq4":
        raise ValueError(f"residual must be 'eq4' or 'eq6', got {residual!r}")
    return {"theta": theta, "omega": omega, "res_theta": dtheta_dt - omega - Kp * Vq - omega_0, "res_omega": domega_dt - Ki * Vq}                      
        