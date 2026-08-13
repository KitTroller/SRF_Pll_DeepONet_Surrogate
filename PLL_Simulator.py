from omegaconf import OmegaConf
import numpy as np
import torch
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent / "config"
pll_constants = OmegaConf.load(CONFIG_DIR / "PLL_Constants.yml")
initial_conditions_config = OmegaConf.load(CONFIG_DIR / "initial_conditions.yml")

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)


class PhysicsEquations():
    def __init__(self, pll_constants=pll_constants):
        """Initialiser loads PLL controler's variables from PLL_Constants.yml"""
        self.Ki = pll_constants.Pll.Ki
        self.Kp = pll_constants.Pll.Kp
        self.omega_0 = 2 * torch.pi * pll_constants.Pll.f_0
        self.v_nominal = pll_constants.Pll.v_nominal
        self.sensors = pll_constants.sensors
        self.time_window = pll_constants.time_window
        self.noise_amplitude = pll_constants.Pll.noise_amplitude
        self.frequency_noise = pll_constants.Pll.frequency_noise

    def park_dqTransform(self, Va, Vb, Vc, theta_pll):
        """Va, Vb, Vc are grid phase voltages and theta_pll is the angle estimate of the PLL, returns the dq coordinates. Also we pick vq with minus sign in front convention"""
        Vd = (2/3) * (Va * torch.cos(theta_pll) + Vb * torch.cos(theta_pll - (2/3) * torch.pi) + Vc * torch.cos(theta_pll + (2/3) * torch.pi))
        Vq = - (2/3) * (Va * torch.sin(theta_pll) + Vb * torch.sin(theta_pll - (2/3) * torch.pi) + Vc * torch.sin(theta_pll + (2/3) * torch.pi))
        return (Vd, Vq)

    def pll_ODEs(self, theta_pll, Vq, omega):
        """We care about dtheta_pll_dt, omega is a made up state variable for the integral"""
        omega_0 = self.omega_0
        dtheta_pll_dt = self.Kp * Vq + omega + omega_0
        domega_dt = self.Ki * Vq
        return (dtheta_pll_dt, domega_dt)
    
    def clarke_alphaBetaTransform(self, Va, Vb, Vc):
        """
        Transforms abc stationary coordinates to alpha-beta stationary coordinates.
        No PLL angle (theta) is required.
        """
        # Using the simplified, equations
        V_alpha = (2/3) * Va - (1/3) * Vb - (1/3) * Vc
        V_beta = (1 / torch.sqrt(torch.tensor(3.0))) * (Vb - Vc)
    
        return (V_alpha, V_beta)

class PLLSimulator():
    def __init__(self, dt = None, initial_conditions=initial_conditions_config):
        self.physics = PhysicsEquations(pll_constants=pll_constants)
        self.n_runs = initial_conditions_config.n_runs
        self.N = self.physics.sensors
        self.dt = dt if dt is not None else self.physics.time_window / self.N
        self.t = (torch.arange(self.N) * self.dt).reshape(1, self.N)
        self.newton_iters = []
        
    def _integrator_step(self, theta_prev, omega_prev, Va, Vb, Vc):
        Vd, Vq = self.physics.park_dqTransform(Va, Vb, Vc, theta_prev)
        theta_k = theta_prev + self.dt * (self.physics.Kp * Vq + omega_prev + self.physics.omega_0)
        omega_k = omega_prev + self.dt * self.physics.Ki * Vq
        return theta_k, omega_k, Vd, Vq    
    
    def _integrator_step_trapezoid(self, theta_prev, omega_prev, Vq_prev, Va_k, Vb_k, Vc_k, tol=1e-10, max_iter=10):
        dt, Kp, Ki = self.dt, self.physics.Kp, self.physics.Ki
        omega_0 = self.physics.omega_0
        
        b = (dt / 2) * (Kp + (dt / 2) * Ki)
        known = theta_prev + dt * (omega_prev + omega_0) + b * Vq_prev
        
        theta = known.clone()
        for it in range(1, max_iter +1):
            Vd, Vq = self.physics.park_dqTransform(Va_k, Vb_k, Vc_k, theta)
            F = theta - known -b * Vq
            dF = 1.0 + b * Vd  # This is the newton method and dVq/dt = -Vd
            step = F / dF
            theta = theta - step
            
            if step.abs().max() < tol:
                break
        self.newton_iters.append(it)
        
        Vd, Vq = self.physics.park_dqTransform(Va_k, Vb_k, Vc_k, theta)
        omega_k = omega_prev + dt/2 * Ki * (Vq_prev + Vq)
        return theta, omega_k, Vd, Vq
    
    def higher_harmonics_noise(self, n_runs, t, omega_g, init_phase):
        h5_amplitude = 0.015 * torch.rand(n_runs, 1) + 0.015  # amplitude range: 1.5% - 3%
        h7_amplitude = 0.012 * torch.rand(n_runs, 1) + 0.008
        h11_amplitude = 0.005 * torch.rand(n_runs, 1) + 0.003
        h13_amplitude = 0.004 * torch.rand(n_runs, 1) + 0.006
        
        time_constant_5 = 0.3 * torch.rand(n_runs, 1) + 0.3  # time constants for harmonics decay
        time_constant_7 = 0.25 * torch.rand(n_runs, 1) + 0.25
        time_constant_11 = 0.2 * torch.rand(n_runs, 1) + 0.2
        time_constant_13 = 0.15 * torch.rand(n_runs, 1) + 0.2
        
        decay_5 = h5_amplitude * torch.exp(-t/time_constant_5)
        decay_7 = h7_amplitude * torch.exp(-t/time_constant_7)
        decay_11 = h11_amplitude * torch.exp(-t/time_constant_11)
        decay_13 = h13_amplitude * torch.exp(-t/time_constant_13)
        
        Va_harmonics = decay_5 * torch.cos(5 * (omega_g * t + init_phase)) + decay_7 * torch.cos(7 * (omega_g * t + init_phase)) + decay_11 * torch.cos(11 * (omega_g * t + init_phase)) + decay_13 * torch.cos(13 * (omega_g * t + init_phase))
        Vb_harmonics = decay_5 * torch.cos(5 * (omega_g * t + init_phase - 2/3 * torch.pi)) + decay_7 * torch.cos(7 * (omega_g * t + init_phase - 2/3 * torch.pi)) + decay_11 * torch.cos(11 * (omega_g * t + init_phase - 2/3 * torch.pi)) + decay_13 * torch.cos(13 * (omega_g * t + init_phase - 2/3 * torch.pi))
        Vc_harmonics = decay_5 * torch.cos(5 * (omega_g * t + init_phase + 2/3 * torch.pi)) + decay_7 * torch.cos(7 * (omega_g * t + init_phase + 2/3 * torch.pi)) + decay_11 * torch.cos(11 * (omega_g * t + init_phase + 2/3 * torch.pi)) + decay_13 * torch.cos(13 * (omega_g * t + init_phase + 2/3 * torch.pi))
        return Va_harmonics, Vb_harmonics, Vc_harmonics

        
    def _grid_phases(self, frequency_offset=0, amplitude_offset=0, init_phase=0):
        """ Shapes are important:
        1) t                  :  (1, N)
        2) noise              :  (3, n_runs, N)
        3) omega_g            :  (n_runs, 1)
        4) init_phase         :  (n_runs, 1)
        5) returns Va, Vb, Vc :  (n_runs, N)
        """
        n_runs = self.n_runs
        N = self.N
        v_nominal = self.physics.v_nominal
        omega_0 = self.physics.omega_0
        # frequency_noise = self.physics.frequency_noise
        noise_amplitude = self.physics.noise_amplitude
            
        t = self.t
            
        noise = noise_amplitude * v_nominal * (torch.rand(3, n_runs, N) - 0.5)
        omega_g = omega_0 + 2 * torch.pi * frequency_offset #frequency_noise * (torch.rand(n_runs, 1) - 0.5)
        #init_phase = torch.linspace(-torch.pi, torch.pi, n_runs).reshape(n_runs, 1)
        
        Va_harmonics, Vb_harmonics, Vc_harmonics = self.higher_harmonics_noise(n_runs, t, omega_g, init_phase)
            
        phase = omega_g * t + init_phase  # (n_runs,1) * (1,N) -> (n_runs,N) broadcasting not matrix multiplication careful bro
        Va = (v_nominal + amplitude_offset) * torch.cos(phase)                  + Va_harmonics + noise[0]
        Vb = (v_nominal + amplitude_offset) * torch.cos(phase - 2/3 * torch.pi) + Vb_harmonics + noise[1]
        Vc = (v_nominal + amplitude_offset) * torch.cos(phase + 2/3 * torch.pi) + Vc_harmonics + noise[2]
            
        return Va, Vb, Vc
        
    def simulate_batch(self, Va, Vb, Vc, theta_init, omega_init, scheme="trapezoid"):
        """ Shapes are important:
        returns theta, omega, Vd, Vq, Valpha, Vbeta :  (n_runs, N)
        theta and omega are the pll state variables 
        Vd and Vq come from park transform 
        and Valpha, Vbeta from clarke transform
        """
        n_runs = self.n_runs
        N = self.N
        theta = torch.zeros(n_runs, N)
        omega = torch.zeros(n_runs, N)
        Vd = torch.zeros(n_runs, N)
        Vq = torch.zeros(n_runs, N)
        
        Valpha = torch.zeros(n_runs, N)
        Vbeta = torch.zeros(n_runs, N)
        
        theta[:, 0] = theta_init
        omega[:, 0] = omega_init
            
        Vd[:, 0], Vq[:, 0] = self.physics.park_dqTransform(Va[:, 0], Vb[:, 0], Vc[:, 0], theta[:, 0])
        
        Valpha, Vbeta = self.physics.clarke_alphaBetaTransform(Va, Vb, Vc)
            
        self.newton_iters = []
            
        for k in range(1, N):
            if scheme == "forward":
                th, om, vd, vq = self._integrator_step(theta[:, k - 1], omega[:, k - 1], Va[:, k - 1], Vb[:, k - 1], Vc[:, k - 1])
            else:
                th, om, vd, vq = self._integrator_step_trapezoid(theta[:, k - 1], omega[:, k - 1], Vq[:, k - 1], Va[:, k], Vb[:, k], Vc[:, k])
            theta[:, k], omega[:, k], Vd[:, k], Vq[:, k] = th, om, vd, vq
        return theta, omega, Vd, Vq, Valpha, Vbeta
    
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    physics = PhysicsEquations()
    
    sim = PLLSimulator()
    print(f"dt = {sim.dt*1e6:.1f} us   N = {sim.N}   "
          f"samples per 20 ms grid cycle = {1/(50*sim.dt):.0f}")

    Va, Vb, Vc = sim._grid_phases()
    print(f"Va shape {tuple(Va.shape)}")
    
    theta0 = torch.zeros(sim.n_runs)      # the PLL always starts guessing zero
    omega0 = torch.zeros(sim.n_runs)

    with torch.no_grad():                 # data generation needs no gradients
        theta, omega, Vd, Vq, Valpha, Vbeta = sim.simulate_batch(Va, Vb, Vc, theta0, omega0, scheme="trapezoid")
        
    def theta_no_pll(Valpha, Vbeta):
        theta = torch.atan2(Vbeta, Valpha)
        theta = torch.Tensor.numpy(theta)
        theta = np.unwrap(theta, axis=1)
        return theta
    atan_theta = theta_no_pll(Valpha, Vbeta)    
    atan_theta = torch.from_numpy(atan_theta)
    t = torch.arange(sim.N) * sim.dt
    tail = slice(int(0.8 * sim.N), None)   # last 20% = settled region

    print(f"\nmean Newton iterations/step : {np.mean(sim.newton_iters):.2f}")
    print(f"settled Vd (want +{sim.physics.v_nominal})    : "f"{Vd[:, tail].mean():.4f} +/- {Vd[:, tail].std():.4f}")
    print(f"settled Vq (want 0)         : "f"{Vq[:, tail].mean():+.2e} +/- {Vq[:, tail].std():.2e}")
    print(f"settled omega_pll [rad/s]   : "f"{omega[:, tail].mean():+.4f} +/- {omega[:, tail].std():.4f}")

    fig, ax = plt.subplots(3, 2, figsize=(9, 7), sharex=True)
    for i in range(0, sim.n_runs, 10):
        ax[0][0].plot(t, Vq[i], lw=0.2)
        ax[1][0].plot(t, Vd[i], lw=0.2)
        ax[2][0].plot(t, omega[i], lw=0.2)
        ax[0][1].plot(t, Va[i], lw=0.2)
        ax[1][1].plot(t, theta[i], lw=0.2)
        ax[2][1].plot(t, atan_theta[i]-theta[i], lw=0.2)
    ax[0][0].set_ylabel("Vq  (phase error)"); ax[0][0].axhline(0, color="k", lw=0.2)
    ax[0][0].set_title("PLL locking: 100 initial phases, every 10th shown")
    ax[1][0].set_ylabel("Vd  (-> +1 pu)")
    ax[2][0].set_ylabel("omega_pll [rad/s]"); ax[2][0].set_xlabel("time [s]")
    ax[0][1].set_ylabel("Va [p.u.]"); ax[0][1].set_xlabel("time [s]")
    ax[1][1].set_ylabel("pll theta [rad]"); ax[1][1].set_xlabel("time [s]")
    ax[2][1].set_ylabel("pll theta - arctan(Vbeta / Valpha) [rad]"); ax[2][1].set_xlabel("time [s]")
    for a in ax:
        a[0].grid(alpha=0.3)
        a[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("pll_lock_check.png", dpi=1024)
    # plt.show()
    print("\nsaved pll_lock_check.png")
