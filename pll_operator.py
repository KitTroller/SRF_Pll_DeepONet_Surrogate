import torch.nn as nn
from omegaconf import OmegaConf
import torch
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent / "config"
model_config = OmegaConf.load(CONFIG_DIR / "DeepONet_models.yml")
initial_conditions_config = OmegaConf.load(CONFIG_DIR / "initial_conditions.yml")
PLL_Constants = OmegaConf.load(CONFIG_DIR / "PLL_Constants.yml")

class MLP(nn.Module):
    def __init__(self, sizes, act=nn.Tanh):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 2):
            layers+=[nn.Linear(sizes[i], sizes[i + 1]), act()]
        layers+=[nn.Linear(sizes[-2], sizes[-1])]
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)
            
            


class Unstacked_DeepONet(nn.Module):
    """output_dim = 1 -> theta only; omega = dtheta/dt - Kp*Vq  (original)
       output_dim = 2 -> [theta, omega]; omega predicted directly (two heads). Trying this out even though we so not care about omega because NN double derivative to find omega was noisy"""

    def __init__(self, model_config=model_config, cfg=None):
        super().__init__()
        if cfg is not None:                        # rebuilding a saved network
            self.hidden_dim   = cfg["hidden_dim"]
            self.output_dim   = cfg["output_dim"]
            self.F            = cfg["F"]
            self.max_freq     = cfg["max_freq"]
            self.trunk_sizes  = list(cfg["trunk_sizes"])
            self.branch_sizes = list(cfg["branch_sizes"])
        else:
            self.hidden_dim = model_config.hidden_dim
            self.output_dim = model_config.output_dim
            self.branch_sizes = list(model_config.sizes.branch_net)
            self.trunk_sizes  = list(model_config.sizes.trunk_net)
            self.max_freq   = model_config.max_fourier_feat_frequency
            self.W = initial_conditions_config.Windows
            self.S = PLL_Constants.sensors
            self.F = model_config.num_fourier_feats
            self.trunk_sizes[0] += 2 * self.F  # F is the number of Fourier Feats and we need 1 + Addition[cos(wt/k),sin(wt/k)] k: ranging [1,F] so 1 + 2F
            self.branch_sizes[0] += 3 * int(self.S / self.W)
            self.trunk_sizes[-1]  = self.hidden_dim
            self.branch_sizes[-1] = self.hidden_dim * self.output_dim   # one block per head because I wanted to
            
        self.trunk_net = MLP(self.trunk_sizes)
        self.branch_net = MLP(self.branch_sizes)

           
    def config(self):
        """Everything load_checkpoint needs. Without this, editing Windows,
        sensors, num_fourier_feats or output_dim silently invalidates every .pth."""
        return {"arch": "Unstacked_DeepONet", "hidden_dim": self.hidden_dim, "output_dim": self.output_dim,
                "F": self.F, "max_freq": self.max_freq, "trunk_sizes": self.trunk_sizes, "branch_sizes": self.branch_sizes}    
        
    def forward(self, branch_input, trunk_input):
        batch_size, num_timesteps, _ = trunk_input.shape
        
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        branch_output = branch_output.view(batch_size, self.output_dim, self.hidden_dim) # 2nd argument: output dimention, 3rd argument: a hidden dimention assumed common among all
        trunk_output = trunk_output.view(batch_size, num_timesteps, self.hidden_dim)
        output = torch.einsum('boh,bth->bot', branch_output, trunk_output)
        return output.transpose(1, 2)
    
        

class Single_PINN(nn.Module):
    """Used as a benchmark"""
    def __init__(self, model_config=model_config, cfg=None):
        super().__init__()
        if cfg is not None:
            self.pinn_sizes = list(cfg["pinn_sizes"])
            self.output_dim, self.F, self.max_freq = cfg["output_dim"], cfg["F"], cfg["max_freq"]
        else:
            self.output_dim = model_config.output_dim
            self.F          = model_config.num_fourier_feats
            self.max_freq   = model_config.max_fourier_feat_frequency
            W, S = initial_conditions_config.Windows, PLL_Constants.sensors
            self.pinn_sizes = list(model_config.sizes.pinn_net)
            self.pinn_sizes[0]  = 3 + 3 * int(S / W) + 1 + 2 * self.F
            self.pinn_sizes[-1] = self.output_dim
        self.pinn_mlp = MLP(self.pinn_sizes)
    
    def config(self):
        return {"arch": "Single_PINN", "pinn_sizes": self.pinn_sizes, "output_dim": self.output_dim, "F": self.F, "max_freq": self.max_freq}
    
    def forward(self, branch_input, trunk_input):
        B, T, _ = trunk_input.shape
        # Tile branch features across trunk points
        branch_tiled = branch_input.unsqueeze(1).expand(B, T, -1)   # (B, T, Bdim)
        x = torch.cat([branch_tiled, trunk_input], dim=-1)          # (B, T, Bdim+Tdim)
        x = x.reshape(B*T, -1)                                      # (B*T, in_dim)
        y = self.pinn_mlp(x)                                        # (B*T, O)
        return y.view(B, T, -1)                                     # (B, T, O)