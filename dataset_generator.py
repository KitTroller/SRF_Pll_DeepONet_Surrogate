from omegaconf import OmegaConf
import torch
import numpy as np
from PLL_Simulator import PLLSimulator, PhysicsEquations
from scipy.stats.qmc import LatinHypercube
import matplotlib.pyplot as plt
import json, os
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent / "config"
initial_conditions_config = OmegaConf.load(CONFIG_DIR / "initial_conditions.yml")

class Dataset_Creator():
    def __init__(self, initial_conditions_config=initial_conditions_config):
        self.init_cond = initial_conditions_config
        self.n_runs = initial_conditions_config.n_runs
        self.pll_simulator = PLLSimulator()
        self.W = initial_conditions_config.Windows # windows
        self.S = int(self.pll_simulator.N//self.W)
        self.t_local = torch.arange(self.S) * self.pll_simulator.dt  # N is the number of sensors 5000 sliced to 10 windows
        self.init_theta_pll_depends_on_init_grid_theta = initial_conditions_config.pll_init_dependent_on_initial_grid_angle
        self.pll_init_angle_range_factor = initial_conditions_config.pll_init_angle_range_factor
        
    def _lhs(self, n, samples):
        return LatinHypercube(d=n).random(samples)
    
        
    def create_initial_condition_space(self, variables=5, total_samples=1000):
        """LHS sampling 5d initial condition grid directly, n_runs points shape of outout (n_runs, 5)"""
        total_samples = self.n_runs
        samples = self._lhs(variables, total_samples)
        init_cond = self.init_cond
        
        low_bound  = np.array([init_cond.ranges.initial_grid_angle[0], init_cond.ranges.frequency_offset[0], init_cond.ranges.amplitude_offset[0], init_cond.ranges.theta_pll[0], init_cond.ranges.omega_pll[0]])
        high_bound = np.array([init_cond.ranges.initial_grid_angle[1], init_cond.ranges.frequency_offset[1], init_cond.ranges.amplitude_offset[1], init_cond.ranges.theta_pll[1], init_cond.ranges.omega_pll[1]])
            
        flag = self.init_theta_pll_depends_on_init_grid_theta
         
        if not flag:
            samples = low_bound + (high_bound - low_bound) * samples
        else: 
            factor = self.pll_init_angle_range_factor
            samples = low_bound + (high_bound - low_bound) * samples
            offset = samples[:, 3] * factor  # making offset range -pi * factor +pi * factor
            samples[:, 3] = samples[:, 0] + offset
            samples[:, 3] = (samples[:, 3] + np.pi) % (2 * np.pi) - np.pi # wrapps thetapll init to -pi, pi again
        return torch.Tensor(samples)

    def solve_ODEs(self, init_conditions):
        """Takes in (n_runs,5) different initial conditions and uses the physics engine to produce a dictionary for all the variables in a 1s time window each of shape (n_runs,N) where N is the number of sensors/samples"""
        pll_simulator = self.pll_simulator
        # COLS = ["initial_grid_angle", "frequency_offset", "amplitude_offset", "theta_pll", "omega_pll"]
        
        Va, Vb, Vc = pll_simulator._grid_phases(init_conditions[:, 1:2], init_conditions[:, 2:3], init_conditions[:, 0:1]) # grid_phases takes as input: frequency_offset, amplitude_offset and init_phase
        theta_pll, omega_pll, Vd, Vq, Valpha, Vbeta = pll_simulator.simulate_batch(Va, Vb, Vc, init_conditions[:, 3], init_conditions[:, 4]) # shape (n_runs, N(time))
        print(f"theta_pll shape for testing: {theta_pll.shape}")
        
        return {
            "theta_pll": theta_pll,
            "omega_pll": omega_pll,
            "Vd":        Vd,
            "Vq":        Vq,
            "Valpha":    Valpha,
            "Vbeta":     Vbeta,
            "Va":        Va,
            "Vb":        Vb,
            "Vc":        Vc,
        }
    
    _F64 = {"theta_pll", "omega_pll", "t_local", "theta0", "omega0", "lhs_samples"}
    _INT = {"run_id", "segment_id"}

    def build_records(self, init_conditions, windowed):
        """Assemble everything that goes in the file. All (n_samples, S) except noted."""
        W = self.W
        records = dict(windowed)                                   # the nine arrays
        records["t_local"]    = self.t_local                       # (S,)   shared
        records["theta0"]     = windowed["theta_pll"][:, 0]        # (n_samples,)
        records["omega0"]     = windowed["omega_pll"][:, 0]
        records["run_id"]     = torch.arange(self.n_runs).repeat_interleave(W)
        records["segment_id"] = torch.arange(W).repeat(self.n_runs)
        records["lhs_samples"] = init_conditions                   # (n_runs, 5)
        return records

    def build_meta(self):
        sim, ph = self.pll_simulator, self.pll_simulator.physics
        return {
            "dt": float(sim.dt), "N": int(sim.N), "W": int(self.W),
            "S": int(sim.N // self.W), "n_runs": int(self.n_runs),
            "Kp": float(ph.Kp), "Ki": float(ph.Ki), "omega_0": float(ph.omega_0),
            "v_nominal": float(ph.v_nominal),
            "noise_amplitude": float(ph.noise_amplitude),
            "columns": ["initial_grid_angle", "frequency_offset", "amplitude_offset", "theta_pll", "omega_pll"],
            "ranges": OmegaConf.to_container(self.init_cond.ranges, resolve=True),
            "forcing_channels": ["Va", "Vb", "Vc"],
        }

    def save_dataset(self, records, meta, path="pll_dataset.npz"):
        out = {}
        for k, v in records.items():
            a = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
            if k in self._INT:
                out[k] = a.astype(np.int32)
            else:
                out[k] = a.astype(np.float64 if k in self._F64 else np.float32)
        out["meta_json"] = np.array(json.dumps(meta))      # savez stores arrays only
        np.savez_compressed(path, **out)
        print(f"saved {path}  {os.path.getsize(path)/1e6:.1f} MB  "
              f"{out['theta_pll'].shape[0]} samples x {out['theta_pll'].shape[1]} points")

    @staticmethod
    def load_dataset(path="pll_dataset.npz", as_torch=True):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(z["meta_json"].item())
        data = {k: (torch.from_numpy(z[k]) if as_torch else z[k])
                for k in z.files if k != "meta_json"}
        return data, meta

    def check_continuity(self, windowed, raw, run_idx=0):
        W = self.W
        stitched = torch.cat([windowed["theta_pll"][run_idx*W + w] for w in range(W)])
        err = (stitched - raw["theta_pll"][run_idx]).abs().max()
        print(f"continuity run {run_idx}: max |err| = {err:.3e}   (want exactly 0)")
        return err

    def generate_dataset(self, path="pll_dataset.npz"):
        init_conditions = self.create_initial_condition_space()
        print(f"initial conditions: {tuple(init_conditions.shape)}")
        raw      = self.solve_ODEs(init_conditions)
        windowed = {k: v.unfold(1, self.S, self.S).reshape(-1, self.S) for k, v in raw.items()}
        self.check_continuity(windowed, raw)
        records = self.build_records(init_conditions, windowed)
        self.save_dataset(records, self.build_meta(), path)
        return records    
        
if __name__ == "__main__":        
    Dataset_Creator().generate_dataset()