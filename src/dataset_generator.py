from omegaconf import OmegaConf
import torch
import numpy as np
from PLL_Simulator import PLLSimulator, PhysicsEquations
from scipy.stats.qmc import LatinHypercube
import matplotlib.pyplot as plt
import json, os
from pathlib import Path

from paths import data as _data

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"   # src/ -> root
initial_conditions_config = OmegaConf.load(CONFIG_DIR / "initial_conditions.yml")

class Dataset_Creator():
    
    def __init__(self, initial_conditions_config=initial_conditions_config, seed=None):
        # seed=None reproduces the historical behaviour EXACTLY (scipy treats seed=None as
        # "draw fresh entropy"), so every dataset made before this existed is unaffected.
        # With a seed, ONE numpy Generator feeds every LatinHypercube call in order, so the
        # IC draw, the fault draw and the gain draw are deterministic but not correlated.
        self.seed = seed
        self._rng = np.random.default_rng(seed) if seed is not None else None
        self.init_cond = initial_conditions_config
        self.n_runs = initial_conditions_config.n_runs
        self.pll_simulator = PLLSimulator()
        self.W = initial_conditions_config.Windows # windows
        self.S = int(self.pll_simulator.N//self.W)
        self.t_local = torch.arange(self.S) * self.pll_simulator.dt  # N is the number of sensors 5000 sliced to 10 windows
        self.init_theta_pll_depends_on_init_grid_theta = initial_conditions_config.pll_init_dependent_on_initial_grid_angle
        self.pll_init_angle_range_factor = initial_conditions_config.pll_init_angle_range_factor
        
        self.disturbances = initial_conditions_config.disturbances
        
    def _lhs(self, n, samples):
        return LatinHypercube(d=n, seed=self._rng).random(samples)
    def create_disturbance_space(self, seed=0):
        """Per-run fault parameters, LHS-sampled like the ICs.

        A run gets a sag, OR a phase jump, OR nothing -- never
        both, so any measured effect is attributable to one mechanism. At
        fraction=0.5 that is 25% sag, 25% jump, 50% clean.

        Returns (n_runs, 5) = [sag_t0, sag_dur, sag_depth, jump_t0, jump_angle_rad]
        and kind (n_runs,) = 0 none / 1 sag / 2 jump.
        """
        n = self.n_runs
        params = torch.zeros(n, 5)
        kind = torch.zeros(n, dtype=torch.long)
        cfg = self.init_cond.get("disturbances", None)
        if cfg is None or not cfg.enabled:
            return params, kind

        # NOT manual_seed(seed): group_split() in train_pll makes the very same call,
        # torch.randperm(n_runs) seeded with split_seed. At seed == split_seed == 0 the
        # two permutations are IDENTICAL, so the validation set perm[:n_val] lands
        # entirely inside the sag block perm[:n_fault//2] -- 750 of 750 val runs were
        # sags, and the model early-stopped on a sag-only validation set.
        # The offset makes the fault assignment independent of any plausible split_seed.
        # self.seed when the family is seeded, else the historical hardcoded 0 -- so a
        # different --lhs_seed gives a genuinely different family, assignment included.
        g = torch.Generator().manual_seed((self.seed if self.seed is not None else seed) + 987_654_321)  
        n_fault = int(round(cfg.fraction * n))
        order = torch.randperm(n, generator=g)
        sag_idx, jump_idx = order[:n_fault // 2], order[n_fault // 2:n_fault]
        lo_t, hi_t = cfg.start

        if len(sag_idx):
            u = torch.as_tensor(self._lhs(3, len(sag_idx)), dtype=params.dtype)
            d_lo, d_hi = cfg.sag.depth
            r_lo, r_hi = cfg.sag.duration
            params[sag_idx, 0] = lo_t + (hi_t - lo_t) * u[:, 0]
            params[sag_idx, 1] = r_lo + (r_hi - r_lo) * u[:, 1]
            params[sag_idx, 2] = d_lo + (d_hi - d_lo) * u[:, 2]
            kind[sag_idx] = 1

        if len(jump_idx):
            u = torch.as_tensor(self._lhs(2, len(jump_idx)), dtype=params.dtype)
            a_lo, a_hi = cfg.phase_jump.angle_deg
            params[jump_idx, 3] = lo_t + (hi_t - lo_t) * u[:, 0]
            params[jump_idx, 4] = torch.deg2rad(a_lo + (a_hi - a_lo) * u[:, 1])
            kind[jump_idx] = 2

        return params, kind
        
    def create_gain_space(self):
        """Per-run (Kp, Ki), LHS-sampled like the ICs. None when disabled.

        PLL_Simulator needs NO change to support this: `_integrator_step_trapezoid`
        computes b = (dt/2)*(Kp + (dt/2)*Ki) and every downstream term broadcasts over
        (n_runs,), so assigning tensors to physics.Kp/Ki is enough. Verified: four runs
        with different gains give four different settling times.
        """
        cfg = self.init_cond.get("gains", None)
        if cfg is None or not cfg.enabled:
            return None
        u = torch.as_tensor(self._lhs(2, self.n_runs), dtype=torch.get_default_dtype())
        kp = cfg.Kp[0] + (cfg.Kp[1] - cfg.Kp[0]) * u[:, 0]
        ki = cfg.Ki[0] + (cfg.Ki[1] - cfg.Ki[0]) * u[:, 1]
        return torch.stack([kp, ki], dim=-1)                      # (n_runs, 2)

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

    def _seed_torch(self):
        """`_grid_phases` draws its sensor noise and harmonic amplitudes with torch.rand,
        so seeding scipy alone would still give a different waveform every run."""
        if self.seed is not None:
            torch.manual_seed(self.seed + 20_260_821)

    def solve_ODEs(self, init_conditions, disturbances=None, gains=None):
        """Takes in (n_runs,5) different initial conditions and uses the physics engine to produce a dictionary for all the variables in a 1s time window each of shape (n_runs,N) where N is the number of sensors/samples"""
        pll_simulator = self.pll_simulator
        if gains is not None:                    # per-run controller gains
            pll_simulator.physics.Kp = gains[:, 0]
            pll_simulator.physics.Ki = gains[:, 1]
        sag = jump = None
        if disturbances is not None:
            p = disturbances                              
            sag  = (p[:, 0:1], p[:, 1:2], p[:, 2:3])      # t0, duration, depth
            jump = (p[:, 3:4], p[:, 4:5])                 # t
        Va, Vb, Vc = pll_simulator._grid_phases(init_conditions[:, 1:2], init_conditions[:, 2:3], init_conditions[:, 0:1], jump=jump, sag=sag) # grid_phases takes as input: frequency_offset, amplitude_offset and init_phase
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

      
    _F64 = {"theta_pll", "omega_pll", "t_local", "theta0", "omega0", "lhs_samples", "kp", "ki"}
    _INT = {"run_id", "segment_id", "fault_kind", "window_faulted"}

    def build_records(self, init_conditions, windowed, disturbance=None, kind=None, gains=None):
        W = self.W
        records = dict(windowed)
        records["t_local"]     = self.t_local
        records["theta0"]      = windowed["theta_pll"][:, 0]
        records["omega0"]      = windowed["omega_pll"][:, 0]
        records["run_id"]      = torch.arange(self.n_runs).repeat_interleave(W)
        records["segment_id"]  = torch.arange(W).repeat(self.n_runs)
        records["lhs_samples"] = init_conditions

        if gains is not None:                    # per-run, broadcast to every window row
            records["kp"] = gains[:, 0].repeat_interleave(W)
            records["ki"] = gains[:, 1].repeat_interleave(W)

        if disturbance is not None:
            S, dt = self.S, self.pll_simulator.dt
            w_start = torch.arange(W) * S * dt                 # (W,)
            w_end = w_start + S * dt
            t0s, durs, t0j = disturbance[:, 0:1], disturbance[:, 1:2], disturbance[:, 3:4]
            # a sag window is one the interval OVERLAPS; a jump window is the one the
            # step falls in. NOTE the PLL transient outruns both, so a window flagged
            # clean immediately after a fault is not truly undisturbed.
            sag_hit  = (kind == 1).unsqueeze(1) & (t0s < w_end) & (t0s + durs > w_start)
            jump_hit = (kind == 2).unsqueeze(1) & (t0j >= w_start) & (t0j < w_end)
            records["disturbance"]    = disturbance                    # (n_runs, 5)
            records["fault_kind"]     = kind.repeat_interleave(W)      # per-run, per-row
            records["window_faulted"] = (sag_hit | jump_hit).reshape(-1).long()
        return records

    def build_meta(self):
        sim, ph = self.pll_simulator, self.pll_simulator.physics
        return {
            "dt": float(sim.dt), "N": int(sim.N), "W": int(self.W),
            "S": int(sim.N // self.W), "n_runs": int(self.n_runs),
            # Kp/Ki become per-run TENSORS when gains are sampled, so there is no single
            # value to record. None is the honest answer; the ranges live under "gains".
            "Kp": None if torch.is_tensor(ph.Kp) else float(ph.Kp),
            "Ki": None if torch.is_tensor(ph.Ki) else float(ph.Ki),
            "omega_0": float(ph.omega_0),
            "v_nominal": float(ph.v_nominal),
            "noise_amplitude": float(ph.noise_amplitude),
            "columns": ["initial_grid_angle", "frequency_offset", "amplitude_offset", "theta_pll", "omega_pll"],
            "ranges": OmegaConf.to_container(self.init_cond.ranges, resolve=True),
            "forcing_channels": ["Va", "Vb", "Vc"], 
            "disturbance_columns": ["sag_t0", "sag_duration", "sag_depth", "jump_t0", "jump_angle_rad"],
            "fault_kind_values": {"0": "none", "1": "sag", "2": "phase_jump"},
            "disturbances": OmegaConf.to_container(self.init_cond.get("disturbances", {}), resolve=True),
            "gains": OmegaConf.to_container(self.init_cond.get("gains", {}), resolve=True),
            "lhs_seed": self.seed,        # None = pre-2026-08-21, UNREPRODUCIBLE
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
        path = _data(path)
        np.savez_compressed(path, **out)
        print(f"saved {path}  {os.path.getsize(path)/1e6:.1f} MB  "
              f"{out['theta_pll'].shape[0]} samples x {out['theta_pll'].shape[1]} points")

    @staticmethod
    def load_dataset(path="pll_dataset.npz", as_torch=True):
        z = np.load(_data(path), allow_pickle=False)
        meta = json.loads(z["meta_json"].item())
        data = {k: (torch.from_numpy(z[k]) if as_torch else z[k])
                for k in z.files if k != "meta_json"}
        if meta.get("slim"):
            # Vd/Vq/Valpha/Vbeta were dropped at save time because they are exact
            # functions of what remains -- rebuild them so a slim file is a drop-in.
            # float64 for the reconstruction, then back to the stored float32 layout.
            ph = PhysicsEquations()
            Va, Vb, Vc = (data[k].double() for k in ("Va", "Vb", "Vc"))
            th = data["theta_pll"].double()
            Vd, Vq = ph.park_dqTransform(Va, Vb, Vc, th)
            Valpha, Vbeta = ph.clarke_alphaBetaTransform(Va, Vb, Vc)
            for k, v in zip(("Vd", "Vq", "Valpha", "Vbeta"), (Vd, Vq, Valpha, Vbeta)):
                data[k] = v.float() if as_torch else v.float().numpy()
        return data, meta

    def check_continuity(self, windowed, raw, run_idx=0):
        W = self.W
        stitched = torch.cat([windowed["theta_pll"][run_idx*W + w] for w in range(W)])
        err = (stitched - raw["theta_pll"][run_idx]).abs().max()
        print(f"continuity run {run_idx}: max |err| = {err:.3e}   (want exactly 0)")
        return err

    def generate_dataset(self, path="pll_dataset.npz"):
        init_conditions = self.create_initial_condition_space()
        disturbance, kind = self.create_disturbance_space()
        gains = self.create_gain_space()
        self._seed_torch()
        raw = self.solve_ODEs(init_conditions, disturbance, gains)
        windowed = {k: v.unfold(1, self.S, self.S).reshape(-1, self.S) for k, v in raw.items()}
        self.check_continuity(windowed, raw)
        records = self.build_records(init_conditions, windowed, disturbance, kind, gains)
        self.save_dataset(records, self.build_meta(), path)
        return records
           
    def generate_multi_W(self, W_list, path_fmt="pll_dataset_W{W}.npz"):
        """One LHS draw, one ODE solve, several windowings. Guarantees the W sweep
        varies ONLY W -- not the initial-condition sample."""
        init_conditions = self.create_initial_condition_space()
        disturbance, kind = self.create_disturbance_space()
        gains = self.create_gain_space()
        print(f"initial conditions: {tuple(init_conditions.shape)}"
              + (f"  gains: {tuple(gains.shape)}" if gains is not None else "  gains: fixed"))
        self._seed_torch()
        raw = self.solve_ODEs(init_conditions, disturbance, gains)        # (n_runs, N); W plays no part
        
        for W in W_list:
            assert self.pll_simulator.N % W == 0, f"W={W} does not divide N={self.pll_simulator.N}"
            self.W = W
            self.S = int(self.pll_simulator.N // W)
            self.t_local = torch.arange(self.S) * self.pll_simulator.dt
            windowed = {k: v.unfold(1, self.S, self.S).reshape(-1, self.S) for k, v in raw.items()}
            self.check_continuity(windowed, raw)
            self.save_dataset(self.build_records(init_conditions, windowed, disturbance, kind, gains), self.build_meta(), path_fmt.format(W=W))
            
if __name__ == "__main__":        
    pass#Dataset_Creator().generate_dataset()