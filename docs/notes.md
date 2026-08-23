# PLL DeepONet — open decisions and deferred changes

Running log of choices made, choices deferred, and the numbers that catch regressions.
Companion to `PLL_DeepONet_BUILD_GUIDE.html`.

---

## START HERE — state as of 2026-08-20

**What this is.** A DeepONet surrogate of an SRF-PLL. Branch takes 3 initial-condition
scalars + `Va,Vb,Vc` over one window; trunk takes `t` (+ Fourier features); two output
heads give `[theta_deviation, omega]`. Applied **recurrently** window-by-window, feeding
its own predicted state forward — 40 handovers over 0.5 s with no ground truth anywhere.
System and benchmark from **Ventura et al.** (`PINNs-in-EMT`, vendored); architecture and
training recipe from **Karampinis et al.** (arXiv:2511.05216). The contribution is the
recurrent whole-window operator, where both papers use a one-step or single-shot map.

**Headline result** — famD, n=5000, 6 seeds, clean runs, 0.5 s, 40 handovers:

| | |
|---|---|
| deployed theta RMS | **3.00e-4 rad = 0.0172 deg**  (6 seeds: [2.58, 3.20]e-4) |
| ... voltage sags / phase jumps | 1.61x / 2.00x clean (ratio bands separate, n=6) |
| vs the trapezoidal solver at the same step | **1.05-1.11x its error at 53x less compute** (batch 1; 2.4x at batch 512 — F41) |
| vs the paper's own NN, inside its trained range | **tie at half the compute**; our network alone **3.4x** more accurate (F40) |
| error growth over 40 handovers | **2.9x**, vs 6.3x for an undamped random walk. Saturates |
| above the noise-driven error floor | **9.6%**, *and the same 9.6% at half the timestep* (F48/F49) |
| trivial `theta0 + w_base*t` baseline | 9.44e-1 rad — a sanity floor, **not** the baseline to lead with |

~45k params, ~2 h to train at n=5000 on an M1 Max.

**Current best config** — and what the sweep programme actually settled:
```
output_dim  2      two heads. Deriving omega from dtheta/dt was the single biggest
                   error source (F10) -- 10x on deployed error
w_phys      0.3    CONFIRMED at n=5000 with F=4: 5.8x val_th / 2.5x operator /
                   1.9x deployed vs w_phys=0. Plateau 0.1-0.6. NOT a scarcity artefact
F           4      Fourier features on -- 4.16x on val_th at 16 seeds (F31)
max_freq    503    empirical. 2*pi/T and w_base BOTH rejected at two window lengths
W           40     12.5 ms. W=40/50/100 are tied; 10 and 20 are worse (F45)
hidden_dim  64     flat from 32 to 128 on the operator; only compounding moves (F46)
sensors     5000   10000 (dt=50us) buys NOTHING on accuracy -- F48's 1.58x was the
                   noise model shrinking with dt, retracted in F49. Move to 10000 only
                   to match the paper's 50 us EMT step
batch_size  512   lr 3e-3   SOAP   patience 40   split_seed 0
```
**The sweep programme's one-sentence result: the architecture, every hyperparameter AND
the timestep are all saturated** — nothing on the list moves the number any more. See
"WHAT ACTUALLY CHANGES", and F49 for why the timestep result had to be retracted.

**The three findings worth leading with**
1. **F10** — deriving `omega = dtheta/dt - Kp*Vq` injects raw sensor noise through `Kp`
   and pins omega to a floor the *formulation* creates. Two heads land 953x below it.
2. **F13/F11** — the physics loss helps 2.4x, but **not by satisfying the physics**:
   the residual moves ~10% while deployed error moves 2.4x. It is **derivative
   (Sobolev) supervision** — the ODE supplies free labels for `dtheta/dt` and
   `domega/dt`. Residuals sit within ~10% of a sensor-noise floor computed in advance.
3. **Gauge invariance** — the physics loss is *exactly* invariant under
   `theta -> theta + a*t + b`, `omega -> omega + a`. Its null space is precisely
   `(theta0, omega0)`, so the data term is the only thing selecting *which* ODE
   solution. This is why the data term cannot be dropped even though physics is 99.9%
   of the loss value. **This is a one-line proof, not a measurement — see below.**
   (It *was* checked numerically to 7 digits by `gauge_check.py`, which has since been
   deleted. Do not cite that file; cite the proof.)

### Why 2 and 3 are the SAME fact, and why both are provable on paper

The strongest version of F13 needs no experiment at all, and it turns "the physics loss
acts as Sobolev supervision" from an inference into a **definition**. The key is what
`Vq` is during training:

```python
# train_pll.run_epoch
out = compute_theta_omega(model, t_query, branch, Vq.unsqueeze(-1), omega_nominal=0.0)
```

`Vq` here is `prep["Vq"]` — the **stored** `Vq` from the dataset, a fixed function of
time, **not** recomputed from the network's own `theta`. So in the loss it is a constant
with respect to the network output. The two residuals are then

```
res_theta = dtheta/dt - omega - Kp*Vq_data          (Vq_data known)
res_omega = domega/dt - Ki*Vq_data                  (Vq_data known)
```

Read the second one plainly: it says *"`domega/dt` must equal this known function of
time."* That is a **supplied label for a derivative** — Sobolev supervision, by
construction, not by interpretation. The first says `dtheta/dt - omega` must equal
another known function. Neither says anything about the *value* of `theta` or `omega`.

Gauge invariance falls straight out. Under `theta -> theta + a*t + b`, `omega -> omega + a`:
`dtheta/dt -> dtheta/dt + a` and `omega -> omega + a`, so `dtheta/dt - omega` is
unchanged; `domega/dt` is unchanged. `Vq_data` does not move because it is data. Both
residuals are therefore *identically* unchanged, and the null space is exactly the two
integration constants `(theta0, omega0)`. Three lines of algebra.

**The important corollary, and it is a limitation, not a boast:** this exact invariance is
a property of the **eq-4 (stored `Vq`) formulation we chose**, not of physics-informed
losses in general. Switch to the eq-6 form — recompute `Vq = park_q(Va,Vb,Vc, theta_pred)`,
which is what *inference* already does — and `Vq` becomes a function of `theta`, the
invariance breaks, and the physics term starts carrying information about *which*
solution. That is open item 6 in the list at the end of this file, and it is now clearly
the most interesting untested variant in the project: it would change the *character* of
the physics loss, not just its weight.

**What exp7 can and cannot add.** The mechanism is settled on paper; exp7 measures the
*magnitude* at n=5000 with F=4. So the claim to make is: "the physics term is derivative
supervision by construction — its null space is exactly the initial conditions — and it
is worth Nx on deployed error at n=5000." Only the N is in doubt.

**F22 — data, not capacity — STANDS, and F46 confirmed it independently.** 5x the data
halved the train/val gap (2.60 -> 1.38) and improved `val_th` 1.83x. F46 then showed
width from 32 to 128 changes the operator not at all, which is the same conclusion
reached from the other side: **capacity was never the binding constraint.**

**Robustness, measured (F42/F43):** grid frequency at **5x** the trained range and
amplitude at **3x** cost under 11%. Sags **deeper and longer** than trained degrade
gracefully (2.6x / 1.3x). The single hard edge is `omega_0`: beyond ~40 rad/s the loop
itself cycle-slips and does not lock inside the 0.5 s window, so there is no settled
trajectory to learn. **Deployable statement: valid for |omega_0| <= 20 rad/s**, which
already covers every realistic initial frequency error.

**Open, in priority order** — see ROADMAP for the full list.
1. **In flight on the cluster:** `w_phys` (result already clear), `fcount` (F=1/2/8 at
   fixed mf — the only test that can move the `max_freq` *mechanism*), `arch`
   (`Single_PINN` — the control Karampinis runs and we never did), `eq6` (Vq from the
   predicted angle), `famE_W40` (separates dt from window length in F48).
2. **DFT of the residual** — offline, no training. The other route to a `max_freq`
   mechanism: does anything physical live at 500-600 rad/s?
3. **Retrain the final model at 10000 sensors, several seeds** once `w_phys` lands.
4. **Unbalanced faults** — the pre-registered prediction that `mf=628` should beat 503,
   because negative sequence lands at 2w = 628 rad/s in dq (roadmap item 8).
5. **Frequency-domain validation**, and the scope caveat to raise with Rahul: ours is a
   standalone/replay emulator, not yet a drop-in EMT co-simulation component.

**The families** — verified from the `data_meta` stored inside the checkpoints, 2026-08-20.
The old "family A/B/C/D" labels are retired: they collided with the `famB`/`famD` filenames
and caused real confusion. Use the filenames.

| family | n_runs | sensors / dt | faults | role |
|---|---|---|---|---|
| `pll_dataset*.npz`, `pll_dataset_W{10,20,40,100}` | 1000 | 5000 / 100 us | no | the originals. All Stage A/B history — **data-starved, superseded** |
| `pll_dataset_n5000_W40.npz` | 5000 | 5000 / 100 us | no | the Stage C re-baseline that proved F22 |
| **`famB_W{10,20,40,100}`** | 5000 | 5000 / 100 us | **no** | the hyperparameter workhorse: Fourier, W, `w_phys`, `fcount`, `hd`, `arch`, `eq6` |
| **`famD_W{40,100}`** | 5000 | 5000 / 100 us | **yes** | **the deployed model.** Every headline number |
| **`famE_W80`** (+ `famE_W40`, derived) | 5000 | **10000 / 50 us** | yes | the dt experiment |

There is **no famC** — that name was burned on the voided disturbance run whose fault
assignment collided with the validation split (F33).

**Generation rule:** always `hpc/generate_family.py`, never `generate_multi_W` directly —
its default `path_fmt` overwrites `pll_dataset_W{W}.npz`, and since `LatinHypercube` is
unseeded an overwritten dataset is **gone for good**, taking every record that names it.
To add a windowing to a family that already exists, use `src/rewindow.py`, which now
**splits and merges** (round trip verified bit-exact) and so preserves the LHS draw.

**Sweep directories**
```
sweeps_wphys/ (20) sweeps_ff/ (10) sweeps_Wtest/ (12) sweeps_ndata/ (4)
        n=1000 history. sweeps_ff has 2 VOID records (mf503 s0/s1, F21).
sweeps_wphys_legacy/ (10)   quarantined, non-comparable, kept for provenance
sweeps_famB_ff/ (48)        Fourier arms, 16 seeds -- the reference control arm that
                            several later experiments copy their w_phys=0.3 / F=4 / h64
                            baseline from, free, instead of retraining it
sweeps_famB_W/ (20)  sweeps_famB_mfW100/ (8)  sweeps_famB_hd/ (4+2)
sweeps_famB_wphys/  sweeps_famB_fcount/  sweeps_famB_arch/  sweeps_famB_eq6/   in flight
sweeps_famD/ (6)     the deployed model, 6 seeds
sweeps_famE/ (2)     dt = 50 us
```

**Traps that have already cost time**
- Seed used to control the train/val **split** as well as init (F16). Now separate:
  `--seed` and `--split_seed`. Always fix `split_seed=0`.
- Checkpoints rebuild from the YAML, so a config edit invalidates every `.pth` unless
  the arch is stored in the checkpoint (it now is, under `cfg`).
- `val_th` is NOT comparable across W — shorter windows shrink the target variance.
  Compare **`rollout_full_rms` at n = W** (always 0.5 s of physical time).
- MPS command-buffer failures corrupt training **without raising**. A divergence guard
  now catches it and writes `status: diverged`; `sweep.load` excludes and names them.
- Memory per process scales as `batch_size * S`. W=40 ~2 GB, W=20 ~3 GB, **W=10 ~5 GB**.
  Keep `batch_size=512` for comparability and reduce parallelism instead.
- Run tags now include the dataset stem and `n_runs`. Before that fix, runs on different
  datasets **overwrote each other's records** — and, worse, each other's **checkpoints**.
  Two `sweeps_ff` records were left pointing at a `.pth` from a different LHS family and
  `reval.py` re-scored them without complaint. See the F21 banner. Audit with
  `scratch/audit.py`-style fingerprinting (compare the checkpoint's stored `mu/sd/s1/s2`
  against `group_split` + `prepare` on the dataset the record names) after any run that
  predates the tag fix.
- **Datasets are unreproducible.** `create_initial_condition_space` calls scipy's
  `LatinHypercube` with **no seed**. Overwrite a `.npz` and every record naming it is
  permanently un-revaluable. `generate_multi_W`'s default `path_fmt` is
  `pll_dataset_W{W}.npz` — exactly the family `sweeps_Wtest` lives on. Use
  `hpc/generate_family.py`, which refuses to overwrite.
- **Runs on different devices are not comparable.** `batches()` shuffles with
  `torch.randperm(n, device=device)`, so CPU, MPS and CUDA draw different minibatch
  orders from the same `--seed`. Never split one comparison across the laptop and the
  cluster.
- **`--n_eval_runs` defaults to 20** in `sweep.py`. Pass 150 on every line. Also note
  that at `n_runs=5000` the validation set is 750 runs, so 150 is no longer "the whole
  validation set" as it was at n=1000 — same standard error as before, not zero.

---

## Settled

| # | decision | choice |
|---|---|---|
| D1 | `Vq` sign convention | minus on the q row (EMT-doc convention). Locks at `eps = 0` with `Vd = +1`. Draft's version locks 180° off. Also flips the Newton Jacobian to `1 + b*Vd`. |
| D3 | sensors per window | 500 (= N/W). Covers the 13th harmonic (650 Hz) with margin. |
| D4 | θ representation | branch gets `(sin θ₀, cos θ₀)` — continuous across the wrap. |
| D6 | forcing channels | `Va, Vb, Vc` — confirmed with Rahul. `Vα/Vβ` and `Vd/Vq` saved too, as free ablations. |
| — | integrator | trapezoidal (implicit, Newton, ~2.3 iters/step), float64. Second-order, verified. |
| — | IC sampling | 5-D Latin Hypercube: grid phase, freq offset, amplitude offset, θ_pll₀, ω_pll₀. |

## Still open — BOTH RESOLVED 2026-08-20, kept for the reasoning

- ~~**D5 — deviation form**~~ **DONE.** `θ̃ = θ − (θ₀ + ω_base·t)`, applied at **load**
  time in `prepare(deviation=True)`, so it stayed A/B-testable as intended. ~94% of the
  signal is a ramp we already know, and it removed a float32 cancellation problem.
  Consequence worth remembering: because the ramp is subtracted, `omega_nominal=0` is
  passed to the residual during training — see the Stage 6 note below.
- ~~**D7 — one-step map (paper eq. 8) vs whole-window operator**~~ **DECIDED: whole-window,
  and it is now the project's contribution.** Both reference papers use a one-step map
  (Ventura eq. 8) or a single-shot trajectory (Karampinis); the recurrent whole-window
  operator is what neither does. Vindicated by measurement: F29 shows their step model
  degrades **9.3x** when forced to a 2x coarser step because its truncation error is tied
  to the step it was trained at, while a window operator carries no such tie.

## Deferred — revisit if training struggles

**1. Window length.** Currently 1 s / 10 windows. Settling is ~0.32 s, so roughly **60% of
windows are post-settling** and carry no transient. Rahul suggests 0.5 s / 5 windows.

Cheapest version: `time_window: 0.5`, `sensors: 2500`. Then `dt` stays 200 µs, `S` stays 500,
branch stays 1503 — **nothing downstream changes.** Post-settling share drops 60% → 20%,
compute halves. This looks like a clear win; do it before the first serious training run.

**2. Correlated PLL / grid initial phase.**
`theta_pll0 = theta_grid0 + U(-pi/2, +pi/2)` instead of two independent draws.

Justification: `eps = ±π/2` is exactly where the pendulum damping term `Kp·V·cos(eps)`
changes sign. Inside that band the loop is always damped; outside it the damping goes
negative and trajectories can run away. So the bound is principled, not arbitrary.

Cost: loses the ~5% cycle-slip cases. But realistically a PLL inside an EMT simulation is
near-locked, and a fault produces a 20–40° phase jump, not 180° — so the narrower range is
arguably *more* representative of deployment.

→ Make the half-width a config parameter so widening it later is a one-line change.

**3. Mid-window phase jumps to model faults.** Currently approximated by random initial grid
phase + random PLL phase (agreed with Rahul). A true mid-window jump is a separate feature.

## Prototyping simplifications — agreed, revisit before any claim

- **Amplitude offset identical on all three phases.** Balanced sags only; no unbalanced
  faults. Unbalance would put a 2ω ripple in `Vq` that an SRF-PLL cannot reject.
- **Harmonics** 5/7/11/13 with decaying envelopes, phase-locked to the fundamental. The
  shift is applied *inside* the multiplication — `cos(h*(ωt + φ − 2π/3))` — so the 5th and
  11th come out correctly **negative sequence** and all four land on 6ω/12ω ripple in dq.
- **Measurement noise** independent per phase (common-mode would cancel exactly in Park).

## Regression numbers — settled region, last 20% of each run

| quantity | LHS ICs (`dataset_generator`) | trivial lock (`PLL_Simulator.__main__`) |
|---|---|---|
| `Vd` | 1.0000 ± **0.0374** | 1.0000 ± **0.0238** |
| `Vq` | ~0 ± **0.0236** | ~0 ± **0.0236** |
| `omega_pll` | ~0 ± **0.726** | ~0 ± **0.014** |
| Newton iters/step | **2.34** | **2.00** |

Where they come from:
- `Vd` spread = measurement noise ⊕ per-run amplitude spread = `√(0.0236² + 0.0289²)`
- `Vq` spread = measurement noise **only** — amplitude cannot enter, because at lock
  `Vq = V·sin(eps) → 0` regardless of `V`
- `omega` spread = grid frequency-offset spread = `2π·0.4/√12`

**Any of these moving means a physics change, not a tuning change.** A one-character bug
(`omega_prev +` missing) once turned the PI into a P-only controller: it still locked, `Vd`
still hit 1.0000, the plots looked perfect — and the only symptom was `Vq` std going
0.0236 → 0.0277.

## Relationship to Rahul's draft — following vs diverging

The architecture is his. The divergences are deliberate and each has a reason I can defend.

**Following his design:**
- Overall pipeline: data generation → operator → physics residual → training → inference
- Two DeepONet variants (stacked / unstacked) plus a plain-MLP baseline
- **Network outputs θ only; ω is derived from ODE 1 via autograd.** His idea, and a good one —
  it makes the kinematic relation exact by construction instead of a penalty term
- Fourier features on the trunk input, built *from* t so autograd still reaches t
- `branch = [initial condition ++ forcing samples]`
- SOAP optimizer, early stopping, self-contained checkpoints
- Harmonic model: 5/7/11/13 with the phase shift applied *inside* the multiplication
  (my first version had it outside, which made the 5th/11th positive sequence — wrong)

**Diverging, with reasons:**

| # | change | why |
|---|---|---|
| 1 | `Vq` sign convention (minus on the q row) | his locks 180° off — verified by simulation, `Vd = −1`. Also flips the Newton Jacobian to `1 + b·Vd`. |
| 2 | trapezoidal instead of backward Euler | his `_implicit_step` is backward Euler = 1st order, no better than explicit here. Trapezoidal is 2nd order at the same ~2.3 Newton iterations, and it is what EMT solvers actually use (TPWRS §II-C). |
| 3 | forcing = `Va,Vb,Vc` instead of `Vq` | `Vq` depends on the θ being predicted, so it is not an independent input function. See the Stage 5 note below — this is what lets the physics loss close the loop. |
| 4 | 5-D LHS initial conditions | his randomises only (θ₀, ω₀); grid phase/amplitude/frequency are fixed or independent. LHS gives N distinct values per variable instead of a handful. |
| 5 | 500 sensors instead of 5000 | 100 kHz sampling of a 650 Hz-max signal. Falls out for free from the window slicing. |
| 6 | D5 deviation form | not in his draft. Removes ~94% of the signal (a known ramp) and a float32 cancellation. |
| 7 | **recurrent rollout test** | not in his draft at all. The surrogate is applied recurrently in a real EMT run, so error compounding is the thing that decides whether this works. Biggest addition. |
| 8 | acceptance tests throughout | not in his draft. Predict a number, not a shape. |

**Bugs found in the draft** (worth mentioning to him):
- `pll_inference.py:17` imports `Train_pll_deeponet`, but the file is `Step_2_Train_pll_deeponet.py` — that script has never run as shipped
- `Step_2_Train_pll_deeponet.py:260-261` builds Adam then immediately overwrites it with SOAP, so `--lr` is silently ignored (PowerDeepONet has the same bug at `training_actions.py:158`)
- the analytic self-test in `pll_physics_deeponet.py` only uses `Vq = 0`, so it cannot catch a wrong `Ki`

## Stage 5 note — a consequence of choosing Va,Vb,Vc

The residual needs `Vq` at collocation times. Two options:

- **(a)** interpolate the *stored* `Vq` (what his draft does). But stored `Vq` was computed from
  the **true** θ → this is the paper's **eq (4)** flavour.
- **(b)** recompute `Vq = park_q(Va, Vb, Vc, θ_pred)` inside the loss. Forcing is external and
  fixed, state comes from the network → the **eq (6)** flavour, and the exact analogue of
  PowerDeepONet's `calculate_from_ode(model, output_col, branch_col)`.

**(b) is more faithful to the closed loop, and is only possible because Va,Vb,Vc are stored.**
Decide explicitly at Stage 5; ideally implement both and ablate.

## Stage 6 note — D5 makes omega_0 cancel out

If the network predicts the **deviation** `θ̃ = θ − (θ₀ + ω_base·t)` rather than θ itself, then
`dθ/dt = dθ̃/dt + ω_base`, and the ω derivation becomes

```
omega = dtheta/dt   - Kp*Vq - omega_base        raw form
      = dtheta~/dt  + omega_base - Kp*Vq - omega_base
      = dtheta~/dt  - Kp*Vq                     deviation form -- omega_base GONE
```

So D5 doesn't just shrink the target — it **removes the ~314 − ~314 subtraction entirely**,
which was the whole numerical argument for it. `compute_theta_omega` needs `omega_nominal`
as a parameter: pass `2*pi*f0` in raw mode, `0.0` in deviation mode.

The residual is unchanged: ω is the same physical quantity either way, so `dω/dt − Ki·Vq`
is identical.

## Experiment log

Validation-set baselines (the "predict zero deviation" predictor):
`var(target_theta) = 0.1660`, `var(target_omega) = 29.96`, `|Ki*Vq| RMS = 66.8 rad/s²`.

| run | optimiser | w_phys | train θ | val θ | gap | R²_θ | resid/term | rollout-10 RMS | best ep |
|---|---|---|---|---|---|---|---|---|---|
| A | Adam 1e-3 | 0 | 3.20e-3 | 5.32e-3 | 1.7× | 0.9680 | 2.670 | 8.758e-02 | 197 (no early stop) |
| B | Adam 1e-3 | 5e-8 | 3.63e-3 | 6.87e-3 | 1.9× | 0.9586 | 1.929 | **7.987e-02** | 160 |
| C | Adam 1e-3 | 1e-3 | 1.76e-3 | 1.64e-2 | **9.3×** | 0.9013 | **0.122** | 2.615e-01 | 50 |

Loss composition at the best epoch (what the optimiser actually saw):
`A: phys/theta = 0` · `B: 0.2` · `C: 37.5`

**Finding F1 — the physics loss trades one-window accuracy for recurrent stability.**
Turning it on made θ **29% worse** and the residual **28% better**, but the 10-window
rollout **8.8% better**. Pure MSE would have called this a regression. The rollout metric
is the one that reflects how the surrogate is actually used.

**Finding F2 — rollout error does not compound over a full run.** Across 1, 2, 5 and 10
windows the *max* error is identical to 4 decimal places (2.22e-01 rad in run B), i.e. peak
error is set by the initial transient and never exceeded. RMS *falls* with more windows
because later windows are settled. Over 1 s of recurrent application the surrogate is stable.

**Finding F3 — rollout error is U-shaped in `w_phys`, with a minimum near 5e-8.**
`0 → 8.76e-2`, `5e-8 → 7.99e-2`, `1e-3 → 2.62e-1`. Too little physics and it is unconstrained;
too much and it sacrifices the data fit that pins down *which* solution you are on.

**Finding F4 — the ODE is learnable, but satisfying it is not the objective.**
At `w_phys=1e-3` the residual drops to 0.122 of the term scale — genuinely satisfied. Yet
R²_θ falls to 0.901 and the rollout triples. Reason: `dω/dt = Ki·Vq` has infinitely many
solutions, one per initial condition. The physics says nothing about *which* trajectory you
are on; only the data term selects it. At `w_phys=1e-3` the optimiser was minimising the
residual 37× harder than the data, and produced a self-consistent model on the wrong path.

**Finding F5 — the physics term currently overfits instead of regularising.**
Train/val gap goes 1.7× → 1.9× → **9.3×** as `w_phys` rises. Cause: the residual is evaluated
at the *training samples' own sensor times, with their own Vq* — a second objective on the
same data, not a constraint on unseen regions. PINNs get their regularisation from
**collocation points**, physics enforced where there are no labels. That is the piece
deferred at Stage 5, and F5 is the measured reason to build it.
`interp1d` in `pll_infer.py` is already the hard part; draw random times in `[0, T_window]`,
interpolate `Va,Vb,Vc` onto them, add that residual. Rahul's draft already has the parameter
(`num_collocation=32`).
*Prediction to test: with collocation the useful `w_phys` should rise by ≥1 order of
magnitude, because the term will finally be regularising rather than competing.*

**Caveat:** 50- and 100-window rollouts are meaningless — a run is only W=10 windows long,
so `rollout` walks into the next runs, each with a different grid phase/amplitude/frequency.
Needs a bounds guard, and a dedicated long-run dataset to test properly.

## 2026-08-12 — post-mortem on the "dependent flag" runs

**Finding F6 — the two runs compared are on different datasets, so the 15× is not a model result.**
`omega0` train-split stats identify the file: `sd=6.0827` (flag OFF) vs `sd=4.4817` (flag ON).
With `pll_init_dependent_on_initial_grid_angle: true` and factor 0.5, the initial phase error is
bounded to **exactly ±π/2** (measured on the saved npz: min −1.570, max +1.568, 0.0% beyond π/2)
and **cycle slips go 0/1000**. Flag OFF gives ε₀ ~ U[−π, π] — half the runs start outside the
easy basin. val θ 6.0e-3 → 3.9e-4 is the task getting easier, not the network getting better.
Both runs also had `num_fourier_feats: 0`, which makes `build_trunk_input` return `t` alone —
so `max_fourier_feat_frequency: 200` was **inert in both**. That run label describes nothing.

**Finding F7 — a single scalar val loss hides where the error is.** Per-window R² on the val
split, best checkpoint (epoch 70):

| window | R²_θ | R²_ω | var(ω_true) | MSE_ω |
|---|---|---|---|---|
| 0 (transient) | 0.994 | **0.986** | 102.43 | 1.445 |
| 1 | 0.976 | 0.973 | 18.23 | 0.491 |
| 2 | 0.965 | 0.588 | 1.06 | 0.438 |
| 4–9 (settled) | ~0.936 | **~0.24** | ~0.50 | ~0.38 |

Window 0 alone carries **67%** of the θ error budget. θ is fine everywhere; ω looks like it
collapses in steady state. It does not — see F8.

**Finding F8 — the settled ω error is a sensor-noise floor, not a training failure.**
The settled ω target is *exactly* `2π·frequency_offset` (corr **0.99992**, sd 0.7257 vs 0.7259) —
one LHS dimension, constant per run. But the prediction is `ω = dθ/dt − Kp·Vq` with the
**measured, noisy** `Vq`, while the target `omega_pll` is the *integrator state*, which is smooth.
So the raw sensor noise enters the prediction through `Kp` and cannot be cancelled:

```
Kp · RMS(Vq_ac)  = 25 × 0.02361 = 0.590 rad/s      floor
model's RMS ω err (settled)     = 0.615 rad/s      measured
floor MSE = Kp²·var(Vq_ac)      = 0.348            vs achieved 0.378  -> 8.6% above floor
```

To carry that term, θ would have to wiggle with amplitude `Kp·A/(2πf)` ≈ **4.4e-4 rad** at
300 Hz — about **30× below θ's own error floor** of 1.0e-2 rad. The θ loss can never see it;
it surfaces only after differentiation. Confirmed spectrally: 88.9% of the ω-error power sits
in 200–2500 Hz while 62% of the θ-error power is at 5–15 Hz.
**So R²_ω ≈ 0.24 in steady state means "at the noise floor", not "broken".** The honest metric
is `MSE_ω − Kp²·var(Vq_ac)`, or report the floor alongside.

**Finding F9 — checkpoints are not portable across a config edit.** `save_checkpoint` records
`arch` but nothing about `num_fourier_feats` or layer sizes; `load_checkpoint` rebuilds from
whatever `DeepONet_models.yml` says at that moment. Editing the YAML silently invalidates every
`.pth` on disk. Separately, `Unstacked_DeepONet.__init__` does
`self.trunk_sizes = model_config.sizes.trunk_net` (an **alias**, not a copy) then `+= 2*F`,
which mutates the shared config: constructing the model three times in one process gives trunk
in-dims **1 → 5 → 9 → 13**. And `build_trunk_input(t, F=model_config.num_fourier_feats, ...)`
freezes `F` in a *default argument* at import time, from its own independent `OmegaConf.load`.
Three separate ways for the trunk builder and the trunk network to disagree.

**Correction to PIPELINE_WALKTHROUGH.html §7.** The suggested experiment (lower
`max_fourier_feat_frequency` 200 → 50) was aimed at the wrong target. The residual is not
spread-spectrum overfitting: it is **53 200 in window 0 vs ~77 in the settled windows**, a 700×
ratio. The residual problem is the transient, not the Fourier budget.

## 2026-08-13 — two output heads. The step change.

Config: W=40 (12.5 ms windows), S=125, F=0, `output_dim: 2`, SOAP on MPS,
batch 512, **`w_phys = 0.0`**, best epoch 140 of 160.

**Finding F10 — deriving omega by differentiating theta was the whole problem.**

| | single head | two heads | |
|---|---|---|---|
| val MSE_omega | 3.816e-1 | **3.654e-4** | **1044x** |
| val RMS omega | 0.618 rad/s | **0.0191 rad/s** | 32x |
| eq-2 residual (common units) | 2.80 | **0.0140** | 200x |
| val MSE_theta | 3.94e-7 | 1.045e-6 | 2.7x worse |
| train/val theta gap | 7.8x | **2.0x** | |
| deployed theta RMS @ 0.5 s | 4.0e-2 | **4.0e-3** | **10x** |
| compounding factor @ 40 windows | 150x | **4.0x** | 37x |

The old omega error sat at 0.3816 against a measured `Kp^2*var(Vq_noise)` floor of
0.348 — it was pinned to a floor that the *formulation* created, not the physics.
Two heads land **953x below** that floor. Diagnosis confirmed end to end.

Figure 4 now *decreases* after n≈5 (6.6e-3 → 4.0e-3 at n=40). The recurrent error is
not merely bounded, it is **contracting**: the surrogate inherited the PLL's own
closed-loop stability, so handover errors get damped faster than they accumulate.

**Finding F11 — with `w_phys = 0`, the physics is already satisfied to the sensor-noise
floor.** The residuals fell 62x (r1) and 72x (r2) from epoch 1 with **no physics term in
the loss at all**. In physical units, against the irreducible floor set by the white
noise in `Vq` (which no smooth function can reproduce):

```
                         achieved      floor     ratio
  eq1  dtheta/dt - omega - Kp*Vq   0.647       0.567    1.14x   [rad/s]
  eq2  domega/dt - Ki*Vq           7.709       6.806    1.13x   [rad/s^2]
```

**Both residuals are within 14% of the best any function could do.** The data term
alone taught the network the ODE. Consequences:
- `w_phys > 0` has ~14% of headroom at labelled points. Expect the sweep to find
  `w_phys = 0` optimal or near it. That is a legitimate result, not a failure.
- The remaining value of a physics term is at **unlabelled** times — i.e. collocation
  is now the only version of the PINN idea with headroom left.
- Reframes F1/F3/F4: the U-shape in `w_phys` was an artifact of the *wrong* residual.

**Finding F12 — the baselines, finally.** Fully recurrent over 0.5 s, 40 handovers:

```
  trivial ramp    theta0 + w_base*t                RMS 9.438e-01 rad
  persistence     theta0 + (w_base + omega0)*t     RMS 3.443e+00 rad
  DeepONet                                         RMS 4.0e-03 rad     236x better
```

0.23 deg of angle error after 0.5 s with no ground truth. Persistence is *worse* than
the plain ramp because carrying a ±20 rad/s omega0 forward diverges while the real PLL
corrects it.

**Caveat — the old W sweep is void.** "W=40 is optimal" was measured under the broken
residual, where the physics term outweighed the theta term by up to 17 800x. Must be
re-run with two heads before it can be claimed.

**Note — figure 6's title is now historical.** It describes the single-head residual.
With two heads nothing is dropped. Relabel as "why the second-order residual was
unusable".

## 2026-08-13 — Stage A: w_phys sweep (W=40, F=0, 1 seed, 10 configs)

| w_phys | roll_rms | per_win | comp | val_th | train_th | r1 | r2 | best_ep / ran |
|---|---|---|---|---|---|---|---|---|
| 0 | 4.003e-3 | 1.010e-3 | 3.96 | 1.045e-6 | 5.339e-7 | 1.230e-2 | 1.398e-2 | 140 / 160 |
| 1e-5 | 4.345e-3 | 9.885e-4 | 4.39 | 1.179e-6 | 5.484e-7 | 1.300e-2 | 1.406e-2 | 158 / 178 |
| 1e-4 | 4.048e-3 | 1.034e-3 | 3.91 | 1.401e-6 | 6.719e-7 | 1.235e-2 | 1.422e-2 | 124 / 144 |
| 1e-3 | 3.416e-3 | 9.680e-4 | 3.53 | 1.069e-6 | 5.222e-7 | 1.219e-2 | 1.401e-2 | 145 / 165 |
| 3e-3 | 3.654e-3 | 8.973e-4 | 4.07 | 8.704e-7 | 4.336e-7 | 1.202e-2 | 1.385e-2 | 152 / 172 |
| 1e-2 | 2.991e-3 | 6.811e-4 | 4.39 | 5.872e-7 | 3.813e-7 | 1.160e-2 | 1.360e-2 | **190 / 200** |
| 3e-2 | 2.680e-3 | 5.730e-4 | 4.68 | 4.386e-7 | 1.135e-7 | 1.126e-2 | 1.324e-2 | **199 / 200** |
| 1e-1 | 2.738e-3 | 6.445e-4 | 4.25 | 4.728e-7 | 1.672e-7 | 1.113e-2 | 1.282e-2 | 169 / 189 |
| 3e-1 | **1.799e-3** | 4.438e-4 | 4.05 | 2.762e-7 | 1.141e-7 | 1.113e-2 | 1.281e-2 | **198 / 200** |
| 1 | 2.352e-3 | 4.600e-4 | 5.11 | 2.745e-7 | 1.918e-7 | 1.111e-2 | 1.279e-2 | **190 / 200** |

**Finding F13 — the physics term helps, but not by satisfying the physics.**
F11 predicted no headroom and therefore no benefit. The premise held, the conclusion
did not. `r1` improves only **10%** and `r2` only **9%** across four decades of
`w_phys` — the residual really is nearly maxed out. Yet `roll_rms` improves **2.2x**
and `per_window_rms` **2.3x**.

Mechanism: **Sobolev / derivative supervision, not constraint enforcement.** The data
terms constrain theta and omega only at the sample points. The residuals additionally
constrain `dtheta/dt` and `domega/dt` — and the ODE supplies those derivative labels
for free from quantities already stored. Supervising the derivative as well as the
value is a known accelerator for function fitting; that is what the physics term is
actually buying here, not equation satisfaction.

Evidence it is **not** classic regularisation: the train/val gap does not shrink
(1.96 -> 2.42, no trend across the sweep). Train and val improve *together*. It helps
optimisation, not generalisation.

**Finding F14 — the sweep independently confirms the F11 noise floor.**
Both residuals asymptote hard at `w_phys >= 0.1` and refuse to move:
`r1: 1.113e-2, 1.113e-2, 1.111e-2` and `r2: 1.282e-2, 1.281e-2, 1.279e-2`.
Measured against the sensor-noise floor:

```
              w_phys = 0        w_phys >= 0.1
  eq1        14.1% above       8.4% above      floor
  eq2        13.3% above       8.3% above      floor
```

Pushing harder buys nothing. Two independent equations stop at the same distance from
a floor computed beforehand from `Kp`/`Ki` times the sensor noise in `Vq`.

**Finding F15 — all the gain is in the operator, none in the rollout stability.**
`compounding` shows no trend (3.53–5.11, mean ~4.2). Since
`roll_rms ~ compounding x per_window_rms` and compounding is flat, the 2.2x is
entirely a better one-window operator. The physics term does not make the recurrence
more stable — the two-head change already did that.

**Noise calibration (a free control).** `w_phys = 1e-5` contributes 1.7% of the loss,
so it is effectively `w_phys = 0`. It differs from the true zero run by **8.5%** on
`roll_rms`. That is the one-seed run-to-run noise: **~±10%**. Therefore:
- 0 -> 1e-4 is flat *within noise*. Correct as predicted.
- 1e-4 -> 3e-1 (2.25x) is far outside noise. **Real.**
- 3e-1 vs 3e-2 (1.49x) is probably real; 3e-1 vs 1.0 (1.31x) is **marginal**.
- **Picking 0.3 as "the optimum" from one seed is NOT justified.** The V at
  0.1 / 0.3 / 1.0 is within what a single seed can manufacture.

**Confound to fix before claiming anything.** Every run with `w_phys >= 0.01` hit the
200-epoch cap with `best_epoch` at 190–199 — still improving when training stopped.
The low-`w_phys` runs converged and early-stopped. The comparison is unfair *in favour
of* high `w_phys`, and the true optimum may be better still. Re-run with
`epochs=600, patience=40`.

**Note:** `val_th` tracks `roll_rms` faithfully here. It misleads across **W** (where
the target variance changes with window length), not across `w_phys` (fixed dataset).
Retitle that panel of figure 07 accordingly.

### Stage A' — same sweep at `epochs=600, patience=40`, 2 seeds. CONCLUSIVE.

| w_phys | seed 0 | seed 1 | mean | range |
|---|---|---|---|---|
| 0 | 4.009e-3 | 3.028e-3 | **3.519e-3** | [3.028, 4.009] |
| 0.03 | 2.429e-3 | 1.755e-3 | **2.092e-3** | [1.755, 2.429] |
| 0.1 | 2.902e-3 | 1.354e-3 | 2.128e-3 | [1.354, 2.902] |
| **0.3** | 1.545e-3 | 1.252e-3 | **1.399e-3** | [1.252, 1.545] |
| 1 | 2.177e-3 | 1.405e-3 | 1.791e-3 | [1.405, 2.177] |

`best_epoch` now 268–474 of 600, so the epoch-cap confound is gone.

**The chain 0 -> 0.03 -> 0.3 separates with no overlap**: `min(w=0) = 3.028 > max(w=0.03)
= 2.429`, and `min(w=0.03) = 1.755 > max(w=0.3) = 1.545`. **2.52x improvement from the
physics term**, on medians, with two seeds. Best single run **1.252e-3 rad = 0.072 deg**;
~675x better than the trivial baseline.

*Supportable:* "w_phys ~ 0.3 is optimal and the physics term buys 2.5x."
*NOT supportable:* "0.3 beats 0.1 and 1" — those overlap.
**Superseded by Stage A'' below — on a fixed split they no longer overlap.**

### Stage A'' — FINAL. Fixed split_seed=0, 600 ep, 2 seeds, 10 points, 20 runs.

> **CONFOUNDED — banner added 2026-08-19 evening. Not withdrawn; re-running.**
>
> Every `w_phys` number in Stage A / A' / A'' — including the **2.43x** and the F13
> "Sobolev supervision" reading — was measured on `pll_dataset.npz`: **n_runs=1000 and
> F=0**. Both are confounds that were controlled for everywhere else in this project:
>
> - **n=1000 is data-starved.** F22/F23 established that, and showed the seed variance
>   which made W=40 unresolvable there was a *scarcity artefact* that vanished at n=5000.
>   A physics term whose mechanism is "free derivative labels" (F13) is exactly the kind
>   of thing that should help **more** when labels are scarce — so the 2.43x may be a
>   scarcity artefact too.
> - **F=0.** F31 then showed Fourier features cut `val_th` 4.16x on their own. If the
>   features supply part of what the physics term was supplying, the two overlap and 0.3
>   is no longer the right weight at the current architecture.
>
> `w_phys=0.3` is nonetheless carried into **all 48 famB runs, both famD models, and
> every number in the Presentation section** — an inherited assumption from a
> data-starved, Fourier-less experiment. It is the last uncontrolled setting in the
> project.
>
> `hpc/exp7_wphys.txt` closes it: 7 values x 4 seeds at n=5000 with F=4/mf=503, plus the
> existing 16-seed `wp0.3` arm copied in free. **Either outcome is a result** — if 2.43x
> survives, the finding is clean for the first time; if it shrinks, the physics loss was
> substituting for data, which *extends* F22 rather than contradicting it and is the
> better sentence.
>
> Until it lands: quote the physics-loss benefit as **"2.4x at n=1000 without Fourier
> features; re-measurement at n=5000 in progress"**, never bare.

| w_phys | seed 0 | seed 1 | mean | per_win | r1 |
|---|---|---|---|---|---|
| 0 | 4.009e-3 | 3.588e-3 | 3.798e-3 | 9.264e-4 | 1.255e-2 |
| 1e-5 | 4.302e-3 | 3.936e-3 | 4.119e-3 | 9.943e-4 | 1.275e-2 |
| 1e-4 | 3.788e-3 | 3.454e-3 | 3.621e-3 | 8.922e-4 | 1.219e-2 |
| 1e-3 | 3.306e-3 | 3.330e-3 | 3.318e-3 | 9.252e-4 | 1.204e-2 |
| 3e-3 | 3.410e-3 | 3.113e-3 | 3.262e-3 | 8.773e-4 | 1.194e-2 |
| 1e-2 | 2.863e-3 | 3.130e-3 | 2.996e-3 | 7.162e-4 | 1.146e-2 |
| 3e-2 | 2.429e-3 | 2.228e-3 | 2.328e-3 | 5.884e-4 | 1.121e-2 |
| 1e-1 | 2.902e-3 | 2.396e-3 | 2.649e-3 | 5.414e-4 | 1.110e-2 |
| **3e-1** | **1.545e-3** | **1.579e-3** | **1.562e-3** | 4.022e-4 | 1.108e-2 |
| 1 | 2.177e-3 | 1.889e-3 | 2.033e-3 | 4.199e-4 | 1.109e-2 |

`w_phys = 0` range [3.588e-3, 4.009e-3]; `w_phys = 0.3` range [1.545e-3, 1.579e-3].
**2.43x, no overlap** — and `max(0.3) = 1.579e-3` also sits below `min(0.03)`, `min(0.1)`
and `min(1)`. With the split fixed, **0.3 clears every neighbour**, so "0.3 is the
optimum" is now defensible rather than "somewhere in 0.03–1".

Seed spread at 0.3 is **2.2%** — the tightest point in the sweep.
`r1` saturates at 1.110 -> 1.108 -> 1.109 from `w_phys >= 0.1`: **F14 confirmed a third
time on clean data.** No amount of physics weight moves the residual off the
sensor-noise floor.

Ten legacy files (5 legacy seed-1 on split 1, 5 at the superseded 200-epoch budget)
quarantined in `sweeps_wphys_legacy/`. `sweeps_wphys/` now holds exactly the
comparable 20.

**Finding F16 — the seed was changing the train/val SPLIT, not just the init.**
`group_split(run_id, val_frac, seed)` used the same `seed` as `torch.manual_seed`, so
each seed drew a different validation set *and* a different 20 runs for
`rollout_metrics`. Seed 1 beat seed 0 in **5 of 5** pairs (ratios 1.32, 1.38, 2.14,
1.23, 1.55) — 1-in-32 under a null of exchangeable seeds. It drew an easier val set.
The w_phys conclusion survives because the monotone chain holds *within* each seed, but
`seed` and `split_seed` are now separate arguments.

Also: the earlier "+/-10% noise" estimate was wrong — it compared `w_phys=1e-5` against
`0` at the *same* seed, so it measured loss sensitivity, not seed variance. True
seed-to-seed spread is **1.23x–2.14x**. One seed per point would have been worthless.

**Why we do NOT drop the data term, even though physics is 99.9% of the loss value.**
At `w_phys = 0.3`: `l_th = 2.4e-7`, `w_omega*l_om = 8.0e-6`, `w_phys*(r1+r2) = 7.1e-3`
-> physics is **99.88%** of the loss. But loss *share* is not gradient *importance*.
The residuals are **invariant to which solution of the ODE the network is on**: if
`(theta, omega)` solves both equations, so does the solution from any other initial
condition, and both score zero. So the physics gradient is exactly **zero** along the
"slide to a different trajectory" direction, and the data term is the only thing with a
non-zero gradient there. Physics constrains the *shape*; data selects *which*
trajectory. This is F4 restated, and it is why the two terms cannot substitute.

*Cheap decisive ablation (one run):* zero both data terms, keep `w_phys`. Prediction:
`r1`/`r2` stay at the floor while `val_th`/`val_om` explode. Would make an
unforgettable slide.

*Research direction worth raising with Rahul:* replace dense supervision with an
**initial-condition loss** only (match `theta(0)=theta0`, `omega(0)=omega0`). Physics +
IC is a classic PINN, and it is the version that matters when dense labels do not exist.

## Stage B — Fourier features (W=40, w_phys=0.3, F=4, 2 seeds, 10 runs)

| config | cycles/window | s0 | s1 | mean | train/val gap | r1 |
|---|---|---|---|---|---|---|
| F=0 control | — | 1.545e-3 | 1.579e-3 | 1.562e-3 | 0.50 | 1.108e-2 |
| mf=100 | 0.20 | 1.490e-3 | 1.690e-3 | 1.590e-3 | 2.41 | 1.085e-2 |
| **mf=503** | **1.00** | 1.121e-3 | 1.068e-3 | **1.095e-3** | **1.15** | 9.656e-3 |
| mf=1885 | 3.75 | 1.452e-3 | 1.624e-3 | 1.538e-3 | 3.16 | **9.400e-3** |
| mf=3770 | 7.50 | 1.140e-3 | 1.950e-3 | 1.545e-3 | 5.00 | 1.002e-2 |

**Finding F17 — the optimal Fourier basis is matched to the WINDOW, not to the signal.**
Control [1.545, 1.579] vs mf=503 [1.068, 1.121]: **1.43x, no overlap.** The prior
hypothesis (match the 300 Hz dq ripple at 1885 rad/s) is **wrong** — 1885 is a null.
503 rad/s is one cycle per 12.5 ms window; with F=4 the ladder is
`503*k/4 = 0.25, 0.50, 0.75, 1.00 cycles` — a quarter-wave Fourier basis on the window.
mf=100 gives 0.05–0.20 cycles: all sub-fundamental, all near-linear, redundant with `t`,
and a null exactly as predicted. mf=1885 and 3770 put 3 of 4 (or 4 of 4) features above
the fundamental, where theta-tilde has no content.
**Why the ripple is irrelevant:** its amplitude in theta-tilde is ~4.4e-4 rad, *below*
the model's ~1e-3 rad error floor. Nothing to gain by representing it.

**Finding F18 — the overfitting gap is minimised at exactly 1.00 cycles per window.**
`0.50 -> 2.41 -> 1.15 -> 3.16 -> 5.00` across 0, 0.20, 1.00, 3.75, 7.50 cycles. Too few
cycles underfits the transient shape; too many memorises the training noise realisation
(5x gap at 7.5 cycles). One cycle is the optimum on both counts.

**Correction to F14.** The `r1` saturation at 1.108e-2 across the whole w_phys sweep was
**not** the sensor-noise floor — it was the trunk basis running out of expressiveness.
With mf=503 the residual reaches 9.656e-3, i.e. **2.1% above the floor** instead of 17%.
(One mf=1885 run hits 9.285e-3, marginally below the estimate — within the ~10%
uncertainty of the moving-average noise/signal split.) F14's *mechanism* stands; the
number 1.11e-2 was a basis limit.

**Third instance of residual != deployed performance:** mf=1885 has the **best** r1
(9.400e-3) and a mediocre rollout; mf=503 has the second-best r1 and the best rollout.

### OPEN QUESTION — is 503 physics, or is it 2*pi/T_window?

```
2*pi / T_window = 502.65 rad/s      vs winning max_freq 503     (0.07% match)
physical candidates: wn 17.3 | Ki 300 | w_base 314.2 | 6w 1885 (tested -> NULL)
```
No physical frequency is within 60% of 503, and the one with real content (1885) is a
null. Circumstantially the basis hypothesis, but untested.

**Decisive test — the two hypotheses disagree when T_window changes:**
- basis: optimum `max_freq` scales as 1/T, so it moves with W
- physics: optimum stays at 503 regardless of W

`W=20 -> 2*pi/T = 251` · `W=40 -> 503` · `W=100 -> 1257`.
**Run W=20 with `max_freq` in {251, 503} x 2 seeds. Four runs, no third outcome.**
A denser sweep around 503 at fixed W=40 CANNOT distinguish them — both predict a peak
there. Wrong experiment; do the W test first.

*If basis wins, the deliverable is a parameter-free rule:* `max_freq = 2*pi/(S*dt)`,
`F = 4` — derived, not tuned, valid at any window length.

### Stage B2 — the W=20 discriminator. Basis hypothesis DISFAVOURED, test incomplete.

Fresh LHS family (`generate_multi_W([10,20,40,100])`), W=20, T=25 ms, F=4, w_phys=0.3,
split_seed=0, 2 seeds, epochs=1200.

| mf | cycles/window | roll_rms mean | range | val_th mean | r1 mean |
|---|---|---|---|---|---|
| 126 | 0.50 | 6.144e-3 | [5.527, 6.761] | 8.08e-6 | 1.204e-2 |
| 251 | **1.00** | 5.356e-3 | [4.875, 5.836] | 6.69e-6 | 1.178e-2 |
| 503 | 2.00 | **5.034e-3** | [4.831, 5.236] | **4.50e-6** | **1.061e-2** |

**Finding F19 — the pure basis hypothesis is disfavoured.** It predicted a peak at
251 (one cycle per window). The curve is instead **monotone improving to 503**, and on
the cleaner metrics it separates: `val_th` 503=[4.361,4.629] vs 251=[5.951,7.435] and
`r1` 503=[1.058,1.064] vs 251=[1.170,1.185] — **no overlap in either**. (`roll_rms`
alone cannot separate 251 from 503; they overlap. val_th and r1 both can, same
direction.)

**BUT THE TEST IS INCOMPLETE — 503 is the top of the tested range at W=20.**
We know "higher is better up to 503"; we do NOT know whether 503 is a peak. At W=40 the
peak was bracketed on both sides (100, 503, 1885, 3770). At W=20 it was only approached
from below. Nothing can be concluded until W=20 is bracketed above.

**A candidate was missed: `w_base = 314.2 rad/s`.** The earlier claim "no physical
frequency within 60% of 503" was wrong to dismiss it — at W=40 the sweep jumped 100 ->
503, a factor of 5, stepping straight over 314. It has never been tested at any W.

This turns out to be lucky. At W=20 the two hypotheses sit on **opposite sides** of
`w_base`:
```
W=40 :  2*pi/T = 503   w_base = 314    (1.6x apart, basis above)
W=20 :  2*pi/T = 251   w_base = 314    (basis now BELOW w_base)
```
So a bracketed W=20 curve separates "peak tracks 2*pi/T" from "peak sits at w_base"
from "peak sits at a fixed 503".

### Stage B2 COMPLETE — W=20 bracketed, 12 runs. Both hypotheses rejected.

| mf | cyc/win | roll_rms range | val_th range | r1 |
|---|---|---|---|---|
| 126 | 0.50 | [5.53, 6.76]e-3 | [7.36, 8.81]e-6 | 1.203e-2 |
| 251 | **1.00** | [4.88, 5.84]e-3 | [5.95, 7.44]e-6 | 1.178e-2 |
| 314 | 1.25 | [4.67, 5.21]e-3 | [6.17, 6.35]e-6 | 1.170e-2 |
| **503** | 2.00 | [4.83, 5.24]e-3 | **[4.36, 4.63]e-6** | 1.061e-2 |
| **754** | 3.00 | [4.18, 5.99]e-3 | **[3.50, 5.15]e-6** | **1.044e-2** |
| 1006 | 4.00 | [5.14, 6.63]e-3 | [5.57, 7.73]e-6 | 1.075e-2 |

**Metric caveat (RESOLVED — see below).** At `n_eval_runs=20` the rollout metric could
not separate 251/314/503/754; seed spreads reached 1.43x. Cause: `rollout_full_rms`
averaged over only 20 of the 150 validation runs, so its standard error was ~11% —
subsampling noise, not training noise.

### The n_eval fix — free, and it settled the question

`reval.py` recomputes rollout metrics for **existing checkpoints** at any `n_eval_runs`,
no retraining (~5 s per checkpoint at W=20). At **n_eval=150 = the full validation set**
there is no subsampling noise left at all.

W=20, n_eval=150:

| mf | cyc/win | seed 0 | seed 1 | mean | spread |
|---|---|---|---|---|---|
| 126 | 0.50 | 5.715e-3 | 5.360e-3 | 5.537e-3 | 1.07 |
| 251 | **1.00** | 5.094e-3 | 4.699e-3 | 4.896e-3 | 1.08 |
| 314 | 1.25 | 4.770e-3 | 4.868e-3 | 4.819e-3 | 1.02 |
| **503** | 2.00 | 4.453e-3 | 4.150e-3 | **4.301e-3** | 1.07 |
| 754 | 3.00 | 4.817e-3 | 3.856e-3 | 4.337e-3 | 1.25 |
| 1006 | 4.00 | 4.443e-3 | 5.068e-3 | 4.755e-3 | 1.14 |

Seed spreads fell from up to 1.43x to **1.02–1.25x**, and the metric now separates:
```
503 vs 126   NO OVERLAP  1.29x
503 vs 251   NO OVERLAP  1.14x    <- 2*pi/T   REJECTED on the deployed metric
503 vs 314   NO OVERLAP  1.12x    <- w_base   REJECTED on the deployed metric
503 vs 754      overlap  1.01x
503 vs 1006     overlap  1.11x
```
**Both hypotheses are now rejected on `roll_rms`, not merely on `val_th`** — closing the
gap flagged above. All three metrics agree.

**Lesson worth keeping: `n_eval_runs=20` was a dominant noise source, not seed variance.**
Standard error scales as 1/sqrt(n_eval). All results re-evaluated at 150; earlier tables
in this file quote n_eval=20 numbers and are noisier versions of the same runs.

### F21 — WITHDRAWN 2026-08-18. The retraction itself was scored on the wrong checkpoints.

> **DO NOT QUOTE THE mf503 ROWS BELOW.** A checkpoint-integrity audit (fingerprinting
> each `.pth`'s stored `mu/sd/s1/s2` against the train-split statistics of every dataset
> on disk) found that exactly 2 of 56 sweep records are scored against a checkpoint that
> is not theirs — and both are the mf=503 arm of this table.
>
> What happened, from file timestamps:
> - **Aug 17 22:52 / 22:58** — `runs/W40_F4_mf503_wp0.3_s{1,0}sp0.pth` were overwritten
>   by the pre-tag-fix runs on `pll_dataset_W40.npz`, the **NEW** LHS family. (This is
>   the same "run tag did not include the dataset" bug recorded in HANDOVER; the damage
>   was wider than the two `sweeps_ndata` duplicates.)
> - **Aug 17 23:38–23:39** — `reval.py sweeps_ff --n_eval 150` re-scored those
>   new-family models against `pll_dataset.npz`, the **OLD** family. Same `W`, same `S`,
>   same `n_runs`, so nothing raised.
>
> The two records still carry their original training fields (`epochs_run 600`,
> `best_epoch 593`, `val_th 1.367e-7`) while the checkpoints on disk hold `737 / 697 /
> 5.575e-8` — `reval.py` only rewrites the rollout fields, which is why the mismatch is
> invisible from inside the record.
>
> **So the W=40 Fourier question has never been measured cleanly at either end.** Stage
> B's numbers came from the right checkpoints but at `n_eval=20`. The `n_eval=150`
> numbers came from the wrong checkpoints. There is no valid measurement to retract
> *or* to defend. Status: **NOT MEASURED**, not "unresolved".
>
> The original Stage B checkpoints are gone. `pll_dataset.npz` still exists, so they
> could be retrained (~1 h each on MPS) — **decided against**: F22 says every n=1000
> number is data-starved, and the n=5000 re-run supersedes the question outright. See
> Stage D.
>
> The other 54 records pass the audit: `sweeps_wphys` (20), `sweeps_Wtest` (12),
> `sweeps_ndata` (4), `sweeps_wphys_legacy` (10), and the 8 non-mf503 `sweeps_ff`
> records. **Nothing else in this file is affected.**

W=40, re-evaluated at n_eval=150:

| cfg | seed 0 | seed 1 | mean | spread | was (n_eval=20) |
|---|---|---|---|---|---|
| F=0 | 1.446e-3 | 1.571e-3 | 1.508e-3 | 1.09 | 1.545 / 1.579 |
| F4 mf100 | 1.518e-3 | 1.564e-3 | 1.541e-3 | 1.03 | 1.490 / 1.690 |
| ~~F4 mf503~~ | ~~1.009e-3~~ | ~~1.666e-3~~ | ~~1.338e-3~~ | ~~1.65~~ | **VOID — wrong ckpt** |
| F4 mf1885 | 1.247e-3 | 1.451e-3 | 1.349e-3 | 1.16 | 1.452 / 1.624 |
| F4 mf3770 | 1.111e-3 | 1.999e-3 | 1.555e-3 | 1.80 | 1.140 / 1.950 |

The mf=1885 and mf=3770 rows are valid and both still overlap F=0, so *nothing* at W=40
separates from the control on `roll_rms` at 2 seeds. That much survives.

**Why W=40 is hard and W=20 is not.** The per-run rollout error is **heavy-tailed**: the
worst validation run is **7–8x the median** in every checkpoint inspected. `roll_rms`
averages per-run RMS, so the tail dominates it. This part is independent of the mf=503
records — mf3770 alone shows a 1.80 spread — and it is confirmed by F24 at n=5000.

**Consequence for resolving W=40:** seed CV is ~25% against a ~13% effect, so `roll_rms`
needs of order 40 seeds. On a laptop that was "not feasible". On the cluster it is one
job array, which is what Stage D does.

### The median statistic — tested, and it does not rescue W=40

`rollout_full_med` (median per-run RMS instead of the mean) was added and everything
re-evaluated. Verdict: **it fixes the tail problem but not the seed problem.**

W=40, median per-run RMS:

| cfg | median mean | median range | spread |
|---|---|---|---|
| F=0 | 9.948e-4 | [9.784e-4, 1.011e-3] | **1.03** |
| F4 mf100 | 1.007e-3 | [1.006e-3, 1.008e-3] | **1.00** |
| ~~F4 mf503~~ | ~~8.672e-4~~ | ~~[6.708e-4, 1.064e-3]~~ | **VOID — wrong ckpt** |
| F4 mf1885 | 9.641e-4 | [8.442e-4, 1.084e-3] | 1.28 |
| F4 mf3770 | 9.909e-4 | [7.647e-4, 1.217e-3] | 1.59 |

The median collapses the spread for F=0 (1.09 -> 1.03) and mf=100 (1.03 -> 1.00), so it
does remove tail-driven noise. **The mf=503 row is void** (see the F21 banner), so the
sentence this section used to end on — "mf=503 keeps a 1.59 spread on the median, which
confirms its seed variance is real training variance" — is withdrawn along with it.

The conclusion survives on other evidence: **mf3770 keeps a 1.59 spread on the median**
too, and F24 shows `roll_rms` holding a 1.66 seed spread at n=5000 *at n_eval=150*. So
"the median fixes the tail but not the seed problem" stands; it just no longer rests on
mf=503.

### The median SHARPENS W=20 — the optimum is 503 specifically

| mf | cyc/win | median mean | median range | spread |
|---|---|---|---|---|
| 126 | 0.50 | 3.937e-3 | [3.726e-3, 4.149e-3] | 1.11 |
| 251 | **1.00** | 3.267e-3 | [3.209e-3, 3.326e-3] | 1.04 |
| 314 | 1.25 | 3.314e-3 | [3.304e-3, 3.325e-3] | 1.01 |
| **503** | 2.00 | **2.982e-3** | [2.774e-3, 3.190e-3] | 1.15 |
| 754 | 3.00 | 3.237e-3 | [2.950e-3, 3.524e-3] | 1.19 |
| 1006 | 4.00 | 3.250e-3 | [3.014e-3, 3.485e-3] | 1.16 |

503 beats 126, **251 and 314 with no overlap on the median as well as the mean** — the
two hypothesis rejections hold on both statistics. And `754` drops from "tied with 503"
(mean) to clearly worse (median 3.237e-3 vs 2.982e-3): its apparent tie was tail-driven.
**On the tail-robust statistic the W=20 optimum is 503 rad/s specifically, not a
503–754 band.**

### What F20 can and cannot claim now

**CAN claim (W=20, both statistics, no overlap):**
- the optimum is **not** at `2*pi/T` (251) -> basis hypothesis REJECTED
- the optimum is **not** at `w_base` (314) -> REJECTED
- the W=20 optimum is **503 rad/s**

**CANNOT claim:** "the optimum sits at the same absolute frequency at both window
lengths." That needs a resolved W=40 optimum, and W=40 has **not been measured** (F21
banner). The fixed-absolute-frequency reading used to be *suggested* by W=40's mean and
median both favouring 503 — **that support is withdrawn**, because both of those W=40
mf=503 numbers came from the wrong checkpoints. It now rests on Stage B's `n_eval=20`
numbers alone, which is no support at all.

**This is the reason `max_freq` is held at 503 for every W in the Stage D sweep** — not
because it is established, but because F20's two *rejections* (2*pi/T and w_base) are
solid at W=20 and a fixed absolute frequency is the only reading left standing. Holding
it fixed makes W the only variable. It is an assumption, and it is labelled as one.

`2*w_base = 628` remains untested. On the median, 754 is now clearly worse than 503, so
the peak lies at or below 754 — which keeps 628 alive but makes 503-itself the better
supported value.

> **Since answered (F31, Stage D):** 628 was tested at 16 seeds and **ties** 503 on
> `val_th` ([0.77, 1.05]e-8 vs [0.77, 1.04]e-8). The peak is a **503–628 plateau**, not a
> point. Choice irrelevant; 503 kept for continuity.

### What we can say about `max_freq` TODAY — and how to close it

**Defensible right now, and it is more than it looks:**
1. The optimum is **not** `2*pi/T_window` (basis/window-relative hypothesis) — rejected at
   W=20 with no overlap on mean *and* median.
2. The optimum is **not** `w_base` = 314 rad/s (grid fundamental) — rejected the same way.
3. It sits on a **503–628 rad/s plateau (80–100 Hz)** at both W=20 (2.0 cycles/window) and
   W=40 (1.0 cycles/window) — i.e. at the **same absolute frequency across a 2x change in
   window length**, which is precisely what a window-relative explanation forbids.

"Two natural hypotheses tested and rejected, optimum bracketed at two window lengths" is
a publishable statement on its own. It does not require knowing *which* physical frequency
it is, and inventing a mechanism that fits is worse than saying this.

**What decides it, in the queue now:** the `mf` sweep at W=100 (exp4) and W=50 (exp3).
- 503 wins again at W=100 (0.40 cyc/win) -> **absolute** confirmed across a 5x range of
  window lengths. Strong result; the mechanism becomes a physics question.
- 1257 (= `2*pi/T` at W=100) wins -> **window-relative** after all, W=40's 503 was a
  coincidence, W=20 was the anomaly — **and the entire W sweep (F32) is confounded**,
  because it held `mf=503` fixed across W by assumption. Would need redoing with `mf`
  tuned per W.

**A third possibility the current design cannot see** — `build_trunk_input` places the
features at `w_k = max_freq * k / F`, so `max_freq` moves the top frequency *and* the comb
spacing together. Every "max_freq optimum" so far is really an optimum over combs.
`hpc/exp8_fcount.txt` separates them: F = 1, 2, 8 at fixed `mf=503`. If **F=1** (the single
frequency 503, nothing else) is as good as F=4, it is one frequency that matters and the
mechanism is physical. If F=1 is clearly worse, it is bandwidth/richness up to 503, and
"why 503" reduces to "why is 80 Hz enough".

**TODO — the cheap diagnostic nobody has run: DFT.** If there really is something at
500–600 rad/s, it should be visible directly in the signals, with no training involved.
Take the residual `theta_pred - theta_true` (and `Vq`, and the per-window target
`theta_deviation`) from a trained model, FFT along the window axis, and average the power
spectrum over validation runs. A peak or a spectral knee near 80–100 Hz would explain the
plateau in one figure; a flat spectrum there would say the optimum is an optimisation
property, not a signal property. Known lines to expect and discount first: 50 Hz
fundamental (314), 100 Hz negative-sequence in dq (628 — the reason 628 was a candidate at
all), and the injected harmonics at 250/350/550/650 Hz. Worth doing regardless of what
W=50/W=100 return, since it is offline and costs one script.

## Stage C — n_runs. THE MOST CONSEQUENTIAL RESULT. We are data-limited.

W=40, F=4, mf=503, w_phys=0.3, split_seed=0, 2 seeds, **equal gradient steps**
(n=1000: 66 batches/ep x ~740 ep = ~49k steps; n=5000: 332 batches/ep x ~175 ep = ~58k).

| | seed | roll_rms | roll_med | per_win | val_th | train_th | **gap** | val_om |
|---|---|---|---|---|---|---|---|---|
| n=1000 | 0 | 7.149e-4 | 5.254e-4 | 2.172e-4 | 5.575e-8 | 2.150e-8 | **2.59** | 4.606e-5 |
| n=1000 | 1 | 1.126e-3 | 9.082e-4 | 2.437e-4 | 7.538e-8 | 2.903e-8 | **2.60** | 1.119e-4 |
| n=5000 | 0 | 9.550e-4 | 8.456e-4 | 1.753e-4 | 3.606e-8 | 2.668e-8 | **1.35** | 8.950e-5 |
| n=5000 | 1 | 5.751e-4 | 4.680e-4 | 1.796e-4 | 3.578e-8 | 2.536e-8 | **1.41** | 3.454e-5 |

| metric | n=1000 | n=5000 | ratio | separation |
|---|---|---|---|---|
| roll_rms | 9.206e-4 | 7.650e-4 | 1.20x | OVERLAP |
| roll_med | 7.168e-4 | 6.568e-4 | 1.09x | OVERLAP |
| **per_window_rms** | 2.305e-4 [2.172,2.437] | **1.775e-4 [1.753,1.796]** | **1.30x** | **NO OVERLAP** |
| **val_th** | 6.557e-8 [5.575,7.538] | **3.592e-8 [3.578,3.606]** | **1.83x** | **NO OVERLAP** |
| val_om | 7.900e-5 | 6.202e-5 | 1.27x | OVERLAP |
| r1 | 9.512e-3 | 9.620e-3 | 0.99x | no overlap (n5000 marginally worse) |

**Finding F22 — the model is DATA-limited, not capacity-limited.**
The train/val gap **halves, 2.60 -> 1.38**. `val_th` improves **1.83x** and
`per_window_rms` **1.30x**, both with no seed overlap. 5x the data, same architecture,
same gradient-step budget. Stop tuning architecture; generate more data.

**Finding F23 — seed variance is a data-scarcity SYMPTOM.** Look at the spreads:
```
val_th   n=1000 spread 1.35   ->   n=5000 spread 1.008   (essentially identical seeds)
per_win  n=1000 spread 1.12   ->   n=5000 spread 1.02
```
**This is why W=40 has been unresolvable.** The 1.59 seed spread that killed the F=0 vs
mf=503 comparison (F21) is a symptom of training on ~850 runs. On 5000 runs the variance
collapses — so **the W=40 Fourier question should be re-run on the big dataset, where it
may well resolve.** Same for any other W=40 comparison that overlapped.

**Finding F24 — `rollout_full_rms` is an intrinsically high-variance metric.** Its seed
spread stayed at ~1.6 even at n=5000, while `val_th` collapsed to 1.008. It compounds
over 40 handovers and is dominated by a heavy tail (worst run 7–8x the median), so it
cannot be tightened by better training. **Use `per_window_rms` / `val_th` to DETECT
differences; report `roll_rms` as the headline with its variance stated honestly.**

`r1` is unchanged at ~9.5e-3 against the 9.459e-3 sensor-noise floor — 0.5–1.7% above it.
More data does not help the residual, because it was already at the floor (F14 again).

**No plot for this.** A two-condition comparison is a table; a chart would only decorate
it. The train/val gap (2.60 -> 1.38) is the whole story.

**Stage C is confirmed at n_eval=150.** These four records were missing the
`n_eval_runs` field (written before `main()` recorded it), which looked like the
`n_eval=20` trap again. `reval.py sweeps_ndata --n_eval 150` changed **nothing** to four
significant figures — they were already at 150. So F24 is real: `roll_rms` holds a 1.66
seed spread at n=5000 *with the subsampling noise removed*. It is training variance and
no amount of evaluation fixes it. **Detect with `val_th` / `per_window_rms`; report
`roll_rms` with its spread stated.**

## Stage D — moving to DTU HPC. Buying SEEDS, not speed.

`docs/HPC workshop.pptx` read; runbook and job scripts in `hpc/`.

**HPC does not make one run faster.** Measured, batch 512, F=4, ms per optimiser step:

| W | S | MPS (M1 Max) | cluster CPU 4 threads |
|---|---|---|---|
| 100 | 50 | 7.5 | 38.9 |
| 40 | 125 | **17.2** | 83.5 |
| 20 | 250 | 31.2 | 117.9 |
| 10 | 500 | 61.1 | 195.9 |

(The 17.2 ms reproduces the 8.1 s/epoch in the Stage C record, so this is calibrated
against a real run, not a synthetic benchmark.) A cluster CPU job is **~4x slower** than
this laptop. A V100 would beat MPS, but not by much and not reliably: the model is 45k
parameters, so a step is ~4 GFLOPs against a V100's ~15 TFLOP/s — the GPU would be
idle ~98% of the step, waiting on kernel launches. **This workload cannot use a big GPU.**

**What HPC actually buys is width.** Total data volume is fixed, so every W costs roughly
the same per run (~25-30 min on MPS). 15 runs serial on the laptop is ~7 h with the
laptop unusable; the same 15 as one LSF array is ~2 h and the laptop stays free.

**And that reframes the whole seed problem.** F21/F24's failure mode was seed variance,
and the answer was always "you'd need ~40 seeds, not feasible". On an array, 8 seeds
costs the same wall-clock as 2. So Stage D spends the width on seeds:

```
exp1  famB_W40   F=0 | F4/mf503 | F4/mf628,  8 seeds each   24 jobs  <- F21 + the last
exp2  famB_W{10,20,100}  F4/mf503,  3 seeds each             9 jobs     mechanism
                         (W=40 comes from exp1)                         candidate
```

`mf=628` (`2*w_base`) is folded into exp1 rather than deferred: it was the last untested
candidate for the 503 optimum, it costs 8 more jobs in an array that was already running,
and testing it *inside* the same 24-run family is strictly better than testing it later
against a different one.

One fresh LHS family `famB_W{10,20,40,100}` at n=5000, so exp2 varies only W, and exp1
sits inside the same family. Both arms of exp1 run on the cluster — the existing MPS
mf=503 records cannot be paired against new CPU F=0 records (device trap above).

`max_freq` is held at 503 across all W in exp2 by assumption, not by result — see
"What F20 can and cannot claim now".

## Stage D RESULT — F21 ANSWERED. The Fourier benefit is real at n=5000.

famB family, W=40, n_runs=5000, w_phys=0.3, split_seed=0, `n_eval_runs=150`, on the
DTU HPC `hpc` queue. **Partial** — the first array lost 34 jobs to an 8 h walltime, so
the Fourier arms are incomplete pending the resubmit. It already separates.

| | n | `val_th` | `per_window_rms` | `roll_rms` |
|---|---|---|---|---|
| F=0 | 16 | [3.00, 4.86]e-8 | [1.71, 2.16]e-4 | [3.66, 4.47]e-4 |
| **mf=503** | 5 | **[0.83, 0.90]e-8** | **[0.87, 0.92]e-4** | **[2.99, 3.22]e-4** |
| mf=628 | 4 | [0.85, 1.05]e-8 | [0.87, 0.98]e-4 | [2.93, 4.80]e-4 |

**Finding F31 — F=0 vs mf=503 separates with NO OVERLAP on all three metrics.**
`val_th` **4.16x** (the intervals are 3.3x apart), `per_window_rms` **2.07x**,
`roll_rms` **1.25x** — and even `roll_rms` separates, the metric F24 called
intrinsically high-variance and which killed this comparison twice.

**F23 is confirmed.** The seed variance that made W=40 unresolvable at n=1000 was a
data-scarcity symptom. At n=5000 it collapsed and the question resolved immediately.
**Stage B's original claim was right**; F21's retraction was an artefact of the
corrupted checkpoints (see the F21 banner), not of the physics.

**mf=503 vs mf=628: tied.** Every interval overlaps. `2*w_base` is neither better nor
worse, so the *mechanism* for the optimum is still open — but it no longer matters for
picking a value. Anything in 503-628 works.

**No epoch-cap confound** (the Stage A trap, checked explicitly): nothing reached the
800 cap. `epochs_run` F=0 367-527, mf503 540-674, mf628 539-554; every run early-stopped
on patience=40. The Fourier arms trained *longer because they kept improving*, not
because they were truncated.

### FINAL at 16 seeds per arm — and the survivorship caveat mattered

| | `val_th` | `per_window_rms` | `roll_rms` |
|---|---|---|---|
| F=0 | [3.00, 4.86]e-8 | [1.71, 2.16]e-4 | [3.66, 4.47]e-4 |
| **mf=503** | **[0.77, 1.04]e-8** | **[0.84, 0.96]e-4** | [2.78, **6.43**]e-4 |
| mf=628 | [0.77, 1.05]e-8 | [0.85, 0.98]e-4 | [2.87, 6.50]e-4 |

**`val_th` 4.16x and `per_window_rms` 2.07x, both with NO OVERLAP at 16 seeds.** That is
F31 and it stands.

**But `roll_rms` NO LONGER separates.** At 5 seeds mf=503 was [2.99, 3.22]e-4 and cleanly
below F=0; at 16 seeds one bad seed takes it to 6.43e-4 and the intervals overlap. The
partial sample was biased exactly as flagged — the survivors of the walltime kill were
the fast-finishing runs. **F24 confirmed a third time: `roll_rms` is heavy-tailed, one
seed in sixteen can swamp it, and it must never be the discriminator.** Medians still
favour mf=503 (3.02e-4 vs 3.79e-4, 1.26x); that is all it supports.

**mf=503 vs mf=628 tied at 16 seeds** ([0.77, 1.04] vs [0.77, 1.05] on `val_th`).
Mechanism still open, choice irrelevant. Keeping 503 for continuity.

## Stage D RESULT 2 — the W sweep. F32.

famB family, n=5000, F=4, mf=503, w_phys=0.3. W=40 taken from the exp1 mf503 arm,
seeds 0-3, so every W has the same seed count. `roll_rms` is the only metric comparable
across W (always 0.5 s of physical time; `val_th` is not — shorter windows shrink the
target variance).

| W | window | `per_window_rms` | `roll_rms` | implied compounding |
|---|---|---|---|---|
| 10 | 50 ms | 1.96e-3 | 3.18e-3 | 1.6 |
| 20 | 25 ms | 3.00e-4 | 6.99e-4 | 2.3 |
| **40** | 12.5 ms | 8.56e-5 | **3.01e-4** [2.78, 3.63] | 3.5 |
| **100** | 5 ms | 3.64e-5 | **3.28e-4** [3.22, 3.31] | 9.0 |

**Finding F32 — the W curve is U-shaped with a flat bottom at W=40-100.** W=40 and
W=100 overlap and are indistinguishable; both beat W=20 by 2.3x and W=10 by 10x.

**The mechanism is in the last column.** Two effects oppose each other:
- shorter windows make the *operator* easier — `per_window_rms` falls **54x** from
  W=10 to W=100, monotonically;
- shorter windows mean *more handovers* — compounding climbs 1.6 -> 9.0.

`roll_rms ~ compounding x per_window`. The operator improves faster than compounding
grows until ~W=40, where they balance. That reproduces the compounding factor of ~4.0
measured independently at W=40 in F10, and it **confirms the pre-registered prediction**
in the old "Level note" that W=40 really does beat W=20 — which could not be claimed
before because the two numbers were on different LHS families.

**W=100 is far more reproducible**: seed spread **1.03x** on `roll_rms` against W=40's
1.31x. If an experiment needs cheap seeds, run it at W=100.

**CAVEAT that limits F32.** `max_freq` was held at 503 for every W *by assumption*.
That is 1.00 cycles per window at W=40 but only **0.40 at W=100** — and the
fixed-absolute-frequency reading (F20) has only ever been tested at W=20 and W=40.
**W=100 may be handicapped by an unsuited basis**, so its tie with W=40 is a lower
bound on how good it could be. Any push past W=100 must re-check `mf` at that W or the
two variables are confounded.

Plot: `graphs/11_sweeps_famB_W_W.png`.

---

## Stage F — disturbances. First run VOID: the fault assignment collided with the split.

**Finding F33 — a seed collision made the validation set 100% sag runs.**
`create_disturbance_space` drew `torch.Generator().manual_seed(seed)` then
`torch.randperm(n_runs)`. `group_split` in `train_pll` makes the **identical call** with
`split_seed`. At `seed == split_seed == 0` the two permutations are the same object of
randomness, so the validation set `perm[:n_val]` sat entirely inside the sag block
`perm[:n_fault//2]`:

```
validation runs that are SAG   : 750 of 750
validation runs that are CLEAN : 0
validation runs that are JUMP  : 0
```

The model was validated, early-stopped and scored on **sags only**, and every clean and
phase-jump run went into training. Fixed by offsetting the fault generator
(`manual_seed(seed + 987_654_321)`), after which the split is 373 clean / 182 sag /
195 jump. **Lesson: two independent random assignments over the same n with the same
seed are not independent.** Third instance of a silent correlation in this project after
F16 (seed controlled the split as well as the init) and F21 (records pointing at the
wrong checkpoints) — all found by checking a distribution that "obviously" had to be
right.

**What the void run still supports** (read as *sag-only*, not as a general result):

| | famD, sag-only val | famB, no faults |
|---|---|---|
| `roll_rms` | 5.43e-4 | 3.02e-4 |
| `per_window_rms` | 1.56e-4 | 8.87e-5 |
| `compounding` | **3.47** | ~3.5 |
| train/val gap | **1.96** | 1.17 |

- **Compounding is unchanged.** Faults make each window harder; they do **not**
  destabilise the recurrence. The surrogate still inherits the loop's contraction.
- **The gap jumped 1.17 -> 1.96, back near the n=1000 regime.** 5000 runs give only
  1250 sags, so *for the fault behaviour* the model is data-starved again — **adding
  disturbances re-opened F22**. The fix is more runs or a higher `fraction`, not more
  capacity. Check this again on the re-run before drawing conclusions.

### Stage F RESULT — 2 seeds, corrected split. The surrogate handles faults.

> **RESOLVED 2026-08-20 — n=6, and the n=2 numbers hold.** `logs/fault_split.log`,
> all six famD checkpoints at `n_eval=150`, split by `fault_kind`:
>
> | subset | median | [min, max] over 6 seeds | spread | vs clean, median |
> |---|---|---|---|---|
> | **clean** | **3.00e-4** | [2.58e-4, 3.20e-4] | 1.24x | — |
> | sag | 4.71e-4 | [4.00e-4, 5.19e-4] | 1.30x | **1.61x** |
> | phase_jump | 5.98e-4 | [5.01e-4, 6.40e-4] | 1.28x | **2.00x** |
> | ALL mixed | 4.57e-4 | [3.83e-4, 4.92e-4] | 1.28x | 1.52x |
>
> **The per-seed RATIOS separate cleanly, which the absolute numbers do not.** sag/clean
> spans [1.41, 1.64] and jump/clean spans [1.68, 2.13] across the six seeds — **no
> overlap**. So "phase jumps are harder for the surrogate than sags" is a real, separated
> finding at n=6, not an artefact of two runs. Pairing within a seed is what buys that:
> the ratio cancels the seed-level offset that inflates the absolute spread to ~1.3x.
>
> Clean compounding across the six: [2.76, 3.16], median 2.89 — this is the number that
> replaced the stale 4.0.
>
> *(Original banner, kept for provenance:)* Seeds 2-5 ran on the laptop
> overnight (`logs/famD_seeds.log`, ~2.1 h each on MPS). **On the laptop deliberately:**
> the two existing seeds are MPS runs, and `batches()` shuffles with
> `torch.randperm(n, device=device)`, so cluster CPU seeds would not be comparable to
> them (the same rule that forced exp1 to re-run *both* arms). Two seeds is the sample
> size that produced F35, which was retracted the same day — the spreads below are 1.00
> to 1.06, which is reassuring but is a spread of two numbers, not a band.
>
> **Where the disturbances came from, since it is easy to muddle.** The fault *types*
> are Ventura et al.'s: their paper tests a 0.965 -> 0.8 pu sag for 25 ms and a 20 deg
> phase jump. What is ours is that the **surrogate's training set contains them** — the
> released network in `PINNs-in-EMT` is trained on clean data, and roadmap item 8 called
> that "the biggest scientific gap". So the claim is not "we invented fault testing", it
> is **"a PLL surrogate that has seen faults during training, scored per fault type"**.
> Our ranges are also a superset of theirs — sag depth 0.5-0.95 pu (vs 0.8), duration
> 20-100 ms (vs 25 ms), jump +/-60 deg (vs 20 deg) — so their test points sit inside our
> training envelope, not outside it.

famD_W40, n=5000, 2500 clean / 1250 sag / 1250 jump, W=40, F=4, mf=503, w_phys=0.3,
`n_eval_runs=150`. Validation split 373 clean / 182 sag / 195 jump.

| subset | seed 0 | seed 1 | spread | vs clean |
|---|---|---|---|---|
| **clean** | **3.0007e-4** | **3.0004e-4** | **1.0001** | — |
| sag | 4.8232e-4 | 4.9275e-4 | 1.02 | 1.61x |
| phase_jump | 6.0326e-4 | 6.3953e-4 | 1.06 | 2.01x |
| ALL mixed | 4.7857e-4 | 4.9246e-4 | 1.03 | 1.60x |

compounding: clean 2.87/2.90 · sag 3.18/3.58 · jump 3.32/3.57 · gap 1.49/1.52

**Finding F34 — training on disturbances costs NOTHING on clean operation.**
Clean `roll_rms` **3.000e-4** vs famB's **3.02e-4**, the model that never saw a fault.
Identical. Half the training data became fault scenarios with no measurable degradation
of normal operation — no catastrophic interference. This is what makes the disturbance
dataset strictly better than famB rather than a trade-off.

**~~F35~~ — RETRACTED THE SAME DAY IT WAS WRITTEN. Do not repeat this claim.**
The draft said: "splitting by fault type made `roll_rms` precise — the clean subset
agrees to four significant figures (3.0007 vs 3.0004e-4) against the 1.66x spread in
Stage C, so the heavy tail was the fault runs all along, and F24 should be amended."
**Both halves are wrong:**
1. **The 1.66x spread was measured on famB, which contains NO faults.** There is no
   mixture there for fault runs to explain. It compared a spread from one dataset to a
   spread from another and invented a link.
2. **It is a variance claim from n=2.** The famB 16-seed data shows `roll_rms` in
   [2.78, 6.43]e-4 with most seeds clustered near 3e-4 and a couple of outliers —
   drawing 2 seeds from that gives a tight pair most of the time. Two points say
   nothing about spread.

**F24 stands unamended.** Whether stratifying by fault type reduces `roll_rms` variance
is **untested**; it would need famD at ~16 seeds to answer. This is the fourth instance
in this project of a conclusion drawn from too few seeds (Stage B's 5, the 4-run
accuracy benchmark, F21 twice) — and it was written *after* "never quote this benchmark
below ~24 runs" went into this file. **The rule is not "be careful", it is: no variance
claim without counting the samples first.**

**What the 2 seeds DO support** (central tendency, not spread): clean `roll_rms`
~3.00e-4 in both, matching famB's 16-seed median of 3.02e-4; and the fault penalties
(1.61x sag, 2.01x jump) reproduce across both seeds.

**Finding F36 — phase jumps are harder than sags (2.01x vs 1.61x vs clean), and the
recurrence stays stable through both.** Compounding rises only 2.9 -> 3.2-3.6. Physical
reading: a phase jump is a step discontinuity in the very quantity being predicted; a
sag changes magnitude, which a locked PLL is comparatively indifferent to (`Vq = V*sin(eps)
-> 0` regardless of V, the same argument as the D-spread note in the regression table).

**On the data question.** Gap 1.49-1.52 against famB's 1.17 — faults do apply some data
pressure, but far less than the **1.96 from the void run** suggested. **n=10000 is NOT
needed.** If more fault headroom is ever wanted, `fraction: 0.8` gives 2000+2000 fault
runs from the same 5000 at zero extra cost — try that before generating anything.

## Stage G — FINAL head to head, both envelopes, famD model. `graphs/13`.

32 runs x 0.5 s, theta RMS against a 12.5 us reference, `src/envelope_figure.py`.
Model: famD seed 0 (trained WITH faults), lean `predict_window`.

| | NEAR-LOCK RMS | FULL RMS | cost [ms/sim-s] |
|---|---|---|---|
| solver @50 us | 6.179e-4 | 6.202e-4 | ~2190 |
| solver @100 us | 8.705e-4 | 8.887e-4 | ~1130 |
| their NN @50 us | **8.848e-4** | 4.908e-3 | 64-69 |
| their NN @100 us | 8.235e-3 | 9.794e-3 | ~32 |
| **ours @100 us** | **8.897e-4** | **9.226e-4** | **20.6** |

**Finding F37 — where both networks are valid they TIE on accuracy, and the cost win is
2x per simulated second but ZERO per output sample.**
Accuracy 8.897e-4 vs 8.848e-4 — **0.6% apart, a tie**. (At n_runs=12 ours looked 5.5%
*better*; that was noise. **Quote the 32-run numbers.**)

**Do NOT quote "3.3x cheaper" from the head_to_head table.** That row (65 ms/sim-s)
includes OUR python driver around their network — the vq recompute, buffer fills and
per-run loop in `paper_nn_at`. Their own optimised inference measured alone
(`paper_pll_only`) is **41.2 ms/sim-s**. The honest numbers:

```
  per simulated second   ours 20.6 ms   theirs 41.2 ms   ->  2.0x
  per output sample      ours  2.06 us  theirs  2.06 us  ->  TIE, exactly
```

**The 2x is entirely the timestep** — we emit one sample per 100 us, they emit one per
50 us, at identical cost per sample. This is F25/F26 confirmed to three significant
figures with the lean path in place, and it is why `sensors`/dt is the real lever.
The defensible claim is **"same accuracy and same per-sample cost, at half the sample
rate"** — not "faster network".

**The test envelope is sized BY MEASUREMENT and cannot be widened.** The flag is now
`their_range` (was the misleading `near_lock` — it is not an edge case, it is the normal
locked operating region). Their scaler says `vq` in +/-0.30. Sweeping the initial phase
error and simulating the FULL trajectory, overshoot included:

| eps_half | omega_half | p99 \|Vq\| | vs their +/-0.30 |
|---|---|---|---|
| **0.05*pi (9 deg)** | 12 | **0.278** | just inside |
| 0.07*pi | 14 | 0.343 | outside |
| 0.09*pi | 15 | 0.370 | outside |

So `0.05*pi` is the **widest** setting that keeps their network inside its trained range.
Widening tests extrapolation, not accuracy. And what fills the budget is the **phase
error, not our sensor noise** — at perfect lock p99|Vq| is 0.061 with our noise and
0.031 without. (Note the overshoot: 9 deg of initial error gives sin(9 deg) = 0.156 at
t=0 but p99 = 0.278 over the trajectory, because the second-order response with
zeta = 0.72 swings past it.)

**Finding F40 — OUR NETWORK IS 3.4x MORE ACCURATE THAN THEIRS. This is the honest
framing of the speed result, and it is better than "2x faster".**
Decomposing each total into network error and discretisation error
(`total^2 = NN^2 + disc^2`, near-lock, 32 runs):

```
  discretisation   @50us  6.179e-4        @100us  8.705e-4
  THEIR network alone     6.333e-4   ->   total 8.848e-4 at 50us
  OUR   network alone     1.838e-4   ->   total 8.897e-4 at 100us
```

**They run a 2x finer step to cover a 3.4x less accurate network. We spend our accuracy
on a 2x coarser step instead — same total error, half the compute.** That is why
"per-sample cost is tied" (F37) is true but misleading on its own: the per-sample cost
being equal is exactly what makes the coarser step a *net win* rather than a wash.

Quotable: **"our operator is 3.4x more accurate than the published step surrogate, which
lets us run at twice the timestep for the same end-to-end error and half the compute."**

*Caveats:* the decomposition assumes the two error sources are independent, and the
12.5 us reference is only 8x finer than 100 us so it carries its own error — the
measured 50->100 us ratio is 1.60 where clean 2nd-order convergence would give 4.0.
Both point the same way but the 3.4x should be quoted as approximate. A finer reference
(or Richardson extrapolation) would tighten it.

**THE OBJECTION THIS WILL DRAW, AND THE ANSWER — always state the two numbers together.**
Their PLL network is **450 parameters**; ours is **45 696**, a factor of ~100. Left alone,
"3.4x more accurate" invites the dismissal *"you used a hundred times the model and five
thousand training trajectories — of course you win."* The answer is F25, and it must be
quoted **in the same breath**, never a section away:

| | params | us per output sample |
|---|---|---|
| their NN | 450 | **1.98** |
| ours (lean) | 45 696 | **1.94** |

**100x the parameters, identical cost per output sample.** Both are dominated by call
overhead (their numpy dispatch, our PyTorch dispatch), not arithmetic. So the extra
capacity is *free on the axis that decides deployment* — it costs no time per sample, and
at 45k float32 weights (~180 kB) it is not a memory obstacle either, microcontroller
included.

That turns the claim from a weak one into a strong one:

> **not** "our bigger model is more accurate" — that is uninteresting
> **but** "at equal cost per sample, they left 3.4x of accuracy unclaimed, and that
> accuracy buys a 2x coarser timestep"

The comparison is therefore fair **on cost**, which is the axis the paper itself argues
on, even though it is not fair on parameter count. State both numbers, concede the
parameter gap openly, and point at the cost column. Conceding it first is what makes the
answer land; letting the reader find the 450 vs 45 696 themselves is what makes it a
wound. The honest residual limitation is the *training* budget — 5000 trajectories vs
whatever they used — which the cost table does not excuse and which should be listed
under limitations, not defended.

**Finding F38 — outside the shared envelope we win 5.3x, and it is a robustness result
not a speed one.** Going near-lock -> full, ours degrades **1.04x** (8.897e-4 ->
9.226e-4) while theirs degrades **5.5x** (8.848e-4 -> 4.908e-3). Their released model
saw `vq` in +/-0.3; the full envelope drives it to +/-1.14, ~4x outside training.
**Careful with the claim**: this is a property of the *trained models*, not of the
architectures — they could retrain on a wider envelope. The defensible sentence is
"their released model is specialised to near-lock", not "their approach cannot do
acquisition". F29 (step-size freedom) is the architectural one; this is not.

**Finding F39 — the network is no longer the binding constraint; the timestep is.**
On the FULL envelope ours sits **3.8% above the solver at the same step** (9.226e-4 vs
8.887e-4). Implied network-only error ~2.0e-4 against a discretisation error of 8.9e-4 —
so dt now dominates by ~4.4x. **This supersedes F27**, which found them equal when the
model was the n=1000 one. Consequence: `sensors`/dt (roadmap 7) is the only remaining
accuracy lever of any size, and it is worth more than the 18% estimated under F27 —
closer to **1.5x** if dt halves.

**F29 reconfirmed at 32 runs:** their NN forced to 100 us degrades **9.3x**
(8.848e-4 -> 8.235e-3). Explicit Euler on a learned derivative is welded to its training
step; a window operator is not. Still the strongest architectural claim we own.

**Figures.** `graphs/13_accuracy_vs_cost_both_envelopes.png` subsumes the old 09/12 pair
— same axes, both envelopes side by side, their extrapolating points ringed in red.

> **FIGURE STATE CORRECTED 2026-08-20.** `graphs/13` was **deleted** on 2026-08-19 and no
> longer exists: it plotted the paper's network on the full envelope, where it is
> extrapolating ~4x outside its trained `vq` range, and showing a method outside its
> training range next to one inside it is not a comparison whichever way it falls.
> Current state: **12** is the sole accuracy-vs-cost figure, restricted to the paper NN's
> trained range; **09** was reduced to a single panel (error vs time) on 2026-08-20 when
> its cost-vs-accuracy panel was found to duplicate 12. Do not cite `graphs/13`.

famD is **1.05-1.11x** the solver at the same step, against 1.21-1.88x for the n=1000
models. The compute ratio is **53x at batch 1**, not the "56x" written elsewhere in this
file — and it must always carry the batch size, because it falls to 2.4x at batch 512
(F41).

**Figure 03 now shows disturbances.** Pass a curated `val_runs` (jump/sag/clean mix) —
no code change needed, `fig_prediction_vs_truth` takes the list. Run 74 is a phase jump:
the left panel shows theta swinging to -0.8 rad with predicted and simulated
indistinguishable, and the right panels show fault onsets as spikes that then **decay**,
i.e. the contraction survives disturbances. All 13 runs stay inside +/-0.004 rad.

## Stage H — cluster harvest, 2026-08-20 morning. Four arrays complete.

### Reading the metrics — what each panel title means

Every sweep figure uses the same four panels, in this order (`plot_sweeps.PANELS`):

| panel title | field | what it is |
|---|---|---|
| "teacher-forced (detects)" | `val_th` | validation MSE on theta, one window, TRUE initial condition. Lowest variance across seeds -> **use this to compare configs** |
| "operator (detects)" | `per_window_rms` | theta RMS, one window, true IC, no chaining. **Operator quality alone** |
| "deployed, tail-robust" | `rollout_full_med` | **median** over runs of each run's RMS, full 0.5 s, 40 handovers, no ground truth |
| "headline (noisy)" | `rollout_full_rms` | **mean** over runs of the same per-run RMS. **The number to quote**, and the one never to pick a winner with |

**`rollout_full_rms` and `rollout_full_med` are the same quantity under two different
statistics** — mean vs median across runs. Not "RMS vs median". They differ only when a
few runs are much worse than the rest, which is exactly F24's heavy tail: one seed in
sixteen swung the mean by 2.3x while the median moved 1.26x.

> **Naming wart, do not let it into the report:** `rollout_full_rms` is a **mean of
> per-run RMS values**, not an RMS of them (`pll_infer.rollout_metrics`: `per(es).mean()`).
> The field name cannot be changed without invalidating every stored record, so describe
> it correctly in prose instead.

**Which to use when.** Deployed is what the surrogate actually delivers, so it is the
headline. But comparisons between configs must be made on `val_th` / `per_window_rms`,
because those separate at n=2-4 seeds while deployed needs 16 and still overlaps (F31).
`compounding = rollout_full_rms / per_window_rms` is the bridge: it isolates what the
handover costs, with the "later windows are easier" effect divided out.

### F44 — `max_freq` at W=100. **1257 does NOT win: F32 is NOT confounded.**

famB_W100, n=5000, F=4, w_phys=0.3, 2 seeds per point, `n_eval=150`.

| mf | cycles/window | `val_th` | `per_window_rms` | `roll_rms` |
|---|---|---|---|---|
| 314 (`w_base`) | 0.25 | [1.55, 1.81]e-9 | [3.95, 4.28]e-5 | [2.52, 4.06]e-4 |
| 503 (incumbent) | 0.40 | [1.29, 1.55]e-9 | [3.62, 3.96]e-5 | **[2.76, 3.25]e-4** |
| **754** | 0.60 | **[1.06, 1.15]e-9** | **[3.25, 3.40]e-5** | [3.33, 3.33]e-4 |
| 1257 (`2*pi/T`) | **1.00** | [1.31, 1.34]e-9 | [3.66, 3.68]e-5 | [3.49, 4.69]e-4 |

**THE ANSWER TO THE QUESTION THAT WAS BLOCKING EVERYTHING.** `2*pi/T` at W=100 is 1257,
and it **ties** 503 on `val_th` (intervals overlap) and **loses** on `roll_rms`. The
window-relative hypothesis is not supported at a third window length. **F32's W sweep,
which held `mf=503` fixed across W by assumption, does NOT need redoing.**

**But the optimum is not perfectly fixed either.** Best `mf` by window length:

| W | window | best mf | cycles/window |
|---|---|---|---|
| 20 | 25 ms | 503 | 2.00 |
| 40 | 12.5 ms | 503-628 | 1.00-1.25 |
| 100 | 5 ms | **754** | 0.60 |

Cycles/window spans 2.00 -> 0.60, so it is **not** window-relative. Absolute frequency
spans 503 -> 754, a factor **1.5** across a **5x** change in window length — where
window-relative would demand a factor of 5. **Absolute is by far the better model, with a
mild upward drift that neither hypothesis predicts.** 754 beats 503, 1257 and 314 on both
low-variance metrics with no overlap — **but at n=2, which is the sample size that
produced the retracted F35. Treat "754 is best at W=100" as provisional; treat "1257 does
not beat 503" as solid, because it is a null and the intervals overlap comfortably.**

### F45 — W sweep with W=50 added. The bottom is **flat from W=40 to W=100**.

`roll_rms` is the only metric comparable across W (always 0.5 s of physical time):

| W | 10 | 20 | **40** | **50** | **100** |
|---|---|---|---|---|---|
| `roll_rms` | 3.198e-3 | 6.988e-4 | **3.010e-4** | 3.265e-4 | 3.276e-4 |

W=40, 50 and 100 all sit inside W=40's own seed band [2.78, 3.63]e-4 — **tied**. W=10 and
W=20 are worse by 10.6x and 2.3x.

> **EPOCH-CAP CAVEAT, found 2026-08-20 — the W=20 and W=10 penalties are UPPER BOUNDS.**
> Epochs actually run, on the cluster CPU nodes, `epochs=800 patience=40`:
>
> | W | S | s/epoch | epochs run | early-stopped? |
> |---|---|---|---|---|
> | 10 | 500 | 55.7 | **800, 800, 800, 800** | **never — capped** |
> | 20 | 250 | 41.2 | **780-800** | **essentially never — capped** |
> | 40 | 125 | 45.8 | 367-800, median **621** | yes |
> | 50 | 100 | 53.7 | 447-585 | yes |
> | 100 | 50 | 59.9 | 382-683 | yes |
>
> **Every W=20 and W=10 model in this project was still improving when training stopped,
> while W=40/50/100 converged and early-stopped.** So the comparison is confounded by
> training budget, not only by window length: 2.3x and 10.6x are the WORST CASE for long
> windows, and the true penalty is smaller by an unknown amount.
>
> This is the same epoch-cap trap the notes flagged for Stage A and explicitly checked for
> in Stage D — and it slipped through here because the W sweep's `epochs_run` was never
> looked at. **Check `epochs_run` against the cap in every future sweep**; it costs one
> column and it is the second time this has nearly cost a finding.
>
> **What does not change:** the *ordering* (longer windows are worse) and the flat bottom
> W=40-100 (all of which early-stopped, so they are clean). What is uncertain is the
> *magnitude* of the W=20 penalty. The exp13/exp14 W=20 arms run at `epochs=1200` and will
> give the first uncapped W=20 numbers — read them before quoting 2.3x anywhere. So W=40 stays, not because it wins but because nothing
in 40-100 differs and it is the incumbent. (`val_th` falls monotonically with W and must
NOT be read across W: shorter windows shrink the target variance.)

### F46 — `hidden_dim` changes the HANDOVER, not the operator.

W=40, famB, 2 seeds each for h32/h128; h64 is the 16-seed `sweeps_famB_ff` arm.

| hidden_dim | `val_th` | `per_window_rms` | `roll_rms` | compounding |
|---|---|---|---|---|
| 32 | [8.30, 10.39]e-9 | [8.77, 9.64]e-5 | [4.85, 5.93]e-4 | **5.5 - 6.2** |
| **64** | [7.7, 10.4]e-9 | [8.4, 9.6]e-5 | [2.78, 6.43]e-4 | ~3.5 |
| 128 | [8.56, 8.99]e-9 | [9.03, 9.22]e-5 | [2.94, 3.23]e-4 | 3.19 - 3.58 |

**`val_th` and `per_window_rms` overlap completely across a 4x range of width.**
One-window operator quality is *flat* from 32 to 128 — so **capacity is NOT the binding
constraint**, contradicting the roadmap's expectation that it would be now that the
train/val gap has closed. F22 stands: data, not capacity.

**What width does buy is recurrent stability.** Compounding degrades to 5.5-6.2 at h32
while single-window accuracy is unchanged, so the narrow network is equally good at one
window and markedly worse at chaining 40 of them. h128 is indistinguishable from h64.
**Keep 64; do not go below it.** (n=2 on the h32/h128 arms.)

### F47 — `sensors` / dt: the lever did NOT appear, and the reason is instructive.

famE = dt 50 us, W=80, S=125 (branch and trunk sizes identical to famB by construction).

| | `val_th` | `per_window_rms` | `roll_rms` | compounding |
|---|---|---|---|---|
| famB W=40, dt=100 us | [7.7, 10.4]e-9 | [8.4, 9.6]e-5 | [2.78, 6.43]e-4 | ~3.5 |
| famE W=80, dt=50 us | [2.43, 2.63]e-9 | [4.73, 4.84]e-5 | [3.11, 4.54]e-4 | 6.6 - 9.4 |

**The operator got 1.9x better per window and 3.6x better on `val_th` — and the deployed
error did not move.** Holding S=125 fixed (to keep the architecture identical) forced W
from 40 to 80, so the 0.5 s rollout now performs **twice as many handovers**. Compounding
roughly doubles and cancels the per-window gain exactly. You cannot hold both the
architecture and the handover count fixed while halving dt; exp6 chose architecture.

### F61 — **NARROWING THE OMEGA RANGE BUYS NOTHING. F59 RETRACTED.**

F59 reported narrow-omega training as 1.2-2.7x better. **That was a validation-set
difficulty artefact, not a model difference**, and it is the same class of error as F45's
epoch cap: each model was scored on **its own family's validation set**, and those sets
are not equally hard. famD's val set contains `omega0` up to +/-20 (cold acquisition);
famH's only goes to +/-2. Comparing `rollout_full_rms` across them compares two different
problems.

Scored on **one common test set** — same ICs, same noise, same protocol, `Kp=25, Ki=300`,
full 0.5 s recurrent rollout, every available seed:

| | omega0 in +/-2 | omega0 in +/-0.2 (his actual band) |
|---|---|---|
| W=40 **wide** (famD, n=6) | 2.394e-4 [2.189, 2.725] | 2.389e-4 [2.172, 2.761] |
| W=40 **narrow** (famH, n=4) | 2.789e-4 [2.411, 3.081] | 2.677e-4 [2.328, 2.805] |
| W=20 **wide** (famI, n=4) | 4.033e-4 [3.592, 4.959] | — |
| W=20 **narrow** (famH, n=4) | 3.941e-4 [3.457, 4.403] | — |

**Every pair overlaps.** At W=40 the wide model's median is *lower*; at W=20 the narrow
one's is, by less than the seed spread. Even in Rahul's tightest band the wide model is at
least as good. **Narrowing the training range is not a lever.**

**Consequences:**
1. **Recommend the WIDE model.** It is statistically identical in his regime *and* covers
   acquisition, so there is no reason to ship a specialist. The "two models with stated
   domains" framing (exp13 header, F60) is unnecessary — there is one model.
2. **F59's omega claim goes with it.** The 1.33-1.66x omega improvement was measured the
   same confounded way. Whether the `sd`-normalisation argument has any force is now
   **untested**, not "half supported".
3. **The `exp13` narrow arms were not wasted** — they are what proved the wide model needs
   no specialisation, which is a stronger and simpler thing to tell him.

**The lesson, and it is the fifth instance this week:** *a metric is only comparable across
models if the thing it is measured on is the same.* `rollout_full_rms` is defined on a
family's own validation split, so it is a **within-family** metric — exactly like
`val_th` across W (F45) and `rollout_full_rms` across dt (F49). **Cross-family claims need
a common test set.** `gain_sensitivity.rollout_err` is now that harness: it synthesises
its own trajectories and needs no dataset, so any two checkpoints can be compared.

### F60 — **THE DELIVERABLE FOR RAHUL.** `graphs/rahul/`, and it changes the F57 advice.

**Gain sensitivity** (`src/gain_sensitivity.py`, `graphs/rahul/03`). Same synthetic test in
every cell — identical ICs, `omega0` in +/-2 (his regime) — sweeping the (Kp, Ki) box and
feeding the gains model those gains as inputs:

| | at their tuning (25, 300) | box median | degradation away from tuning |
|---|---|---|---|
| **fixed Kp/Ki** (famH_W40) | **2.851e-4** | 5.444e-2 | **191x** |
| **Kp/Ki as inputs** (famK_W40) | 7.216e-4 | 1.107e-3 | **1.53x** |

**This is the argument for the feature, and it is not the one F57 made.** At his exact
tuning the fixed model is **2.5x better** — so if his gains never move, use it. But one
grid step away (Ki 300 -> 200) the fixed model is **38x worse** than the tunable one
(2.53e-2 vs 6.69e-4), and over the box it is **49x** worse. A model trained at one tuning
is not approximately right at a nearby tuning; it is useless.

**And it revises F57's headline.** The 3.5x average cost of tunability was dragged up by
the **underdamped corner** (Kp=10, zeta ~ 0.20) — the whole Kp=10 column runs 2.4e-3 to
8.7e-3 while everything at Kp >= 18 sits between 6e-4 and 2.4e-3. **At his operating point
the cost is 2.5x, not 3.5x.** Quote 2.5x to him; quote 3.5x only as the average over a box
spanning zeta 0.20-2.50, which is far wider than he needs.

**Speed, measured on the laptop at batch 1** (per-call timing of `predict_window`,
200 calls each):

| model | calls/sim-s = 1/(S*dt) | **ms/sim-s** | us/call |
|---|---|---|---|
| W=40 fixed | 80 | **19.4** | 241.9 |
| **W=20 fixed** | 40 | **10.9** | 272.7 |
| W=40 gains | 80 | 20.6 | 257.2 |
| W=20 gains | 40 | 11.2 | 279.7 |

**F25 confirmed directly:** doubling the samples per window raises the per-call cost only
**13%** (241.9 -> 272.7 us), because inference is overhead-bound rather than FLOP-bound.
So halving the call rate gives **1.78x** less compute, close to the ideal 2x. And **making
Kp/Ki inputs costs ~5% of inference time** — the accuracy hit is the only real price.

**THE RECOMMENDATION TO SEND HIM:**
1. Gains fixed at 25/300 -> **W=40 narrow fixed**, 2.851e-4 rad at 19.4 ms/sim-s.
2. Wants half the compute -> **W=20 narrow fixed**, 10.9 ms/sim-s. Note the combination
   beats either lever: W=20 *plus* a narrow omega range costs only **1.41x** against his
   current W=40 wide model, where W=20 alone costs 1.88x.
3. Wants to retune Kp/Ki without a retrain -> **the gains model**, 2.5x on accuracy and
   ~5% on speed. Stay above Kp ~ 18; the underdamped corner is where it falls apart.

**Ask him two things before any of this is final:** what `sim_step` he runs at (everything
above is dt = 100 us; their solver defaults to 50 us), and how far he actually needs to
retune. A gain box of +/-20% around 25/300 instead of our zeta 0.20-2.50 span would fit far
better and shrink the 2.5x.

### F57 — **Kp/Ki as inputs costs 3.5x deployed averaged over the gain box.** `exp14`, and it is the
number to give Rahul.

Same omega range, same W, same everything — differs only by whether the controller gains
are sampled per run and fed to the branch:

| comparison | fixed-gain deployed | GAINS deployed | cost | `val_th` cost (MSE) | gap: fixed -> gains |
|---|---|---|---|---|---|
| W=40, narrow omega (famH -> famK) | 3.956e-4 | 1.386e-3 | **3.50x** | 17.7x | 1.17 -> 1.47 |
| W=40, wide omega (famD -> famJ) | 4.580e-4 | 1.645e-3 | **3.59x** | 9.2x | 1.44 -> 1.47 |
| W=20, wide omega (famI -> famJ) | 8.589e-4 | 3.035e-3 | **3.53x** | 13.3x | 2.40 -> 2.48 |

**3.50, 3.59, 3.53 — the cost is the same to within 3% across two window lengths and two
omega ranges.** That is a remarkably stable price for the flexibility, and it is exactly
what Rahul needs in order to decide: *"you can retune Kp and Ki without a retrain, and it
costs 3.5x on deployed angle error."* In his co-simulation the fixed model agreed to
~1e-3 rad, so a gains model would land near 3.5e-3 rad — still 0.2 degrees.

**It is NOT a data-starvation problem, so do not throw runs at it.** The train/val gap
barely moves (1.44 -> 1.47 at W=40 wide, 2.40 -> 2.48 at W=20). The model is not
overfitting the bigger input space; it is simply solving a harder problem — a **family**
of controllers spanning zeta = 0.20 to 2.50 — on the same budget. Consistent with F55.
**The lever, if 3.5x is too much, is a NARROWER gain range, not more data.** Ask Rahul how
far he actually needs to retune; +/-20% around 25/300 would be a far easier fit than the
2.5x-in-zeta span we sampled.

### F58 — the F45 epoch cap RESOLVED, and the W=20 penalty shrinks.

The exp13/exp14 W=20 arms ran at `epochs=1200` and **finally early-stopped**: famI_W20 at
783-909 (best 743-869), famH_W20 at 784-981 (best 744-941). So the old 800-epoch W=20 runs
were being cut off right at convergence, exactly as F45's caveat suspected.

Properly trained, W=20 vs W=40 on deployed error is **1.88x** (8.589e-4 vs 4.580e-4), not
the **2.3x** F45 reported from capped runs. **Use 1.88x.** *(Caveat: famI is a different
LHS family from famD, so this is not a within-family comparison — but it is the only
uncapped W=20 measurement in existence, and it moves in the direction the caveat
predicted.)*

**Still capped:** the W=40 arms of exp13/exp14 (`epochs=800`, several hitting 800 with
best at 777-799). Those numbers are also slight underestimates. **Check `epochs_run`
against the cap before quoting anything** — that is now four separate occasions.

### F59 — the narrow-omega prediction: HALF right, and the half that fails is mine.

`exp13` tested the pre-registered claim that narrowing `omega_pll` to +/-2 would improve
**omega a lot and theta modestly**, because `sd` drops from ~5.3 to ~0.5 and the omega0
branch channel starts carrying information (see the exp13 header):

| | theta improvement | omega improvement | prediction? |
|---|---|---|---|
| W=20 (famI -> famH, both fresh cluster generates) | 1.48x | **2.74x** | **holds** |
| W=40 (famD -> famH_W40, different families/machines) | **2.01x** | 1.78x | **fails** |

**The controlled comparison supports it; the confounded one does not.** famI_W20 and
famH_W20 were generated the same day on the same machine and differ only in the omega
range — there omega improves 1.9x more than theta, as predicted. famD is a laptop family
from days earlier, so the W=40 row crosses LHS draws *and* machines and is weaker evidence
either way.

**State it as:** *"narrowing the omega range improves everything by 1.5-2.7x, and in the
one controlled comparison the improvement is concentrated in omega as the normalisation
argument predicts — but a second, confounded comparison does not reproduce that split, so
the mechanism is suggested rather than established."* The **deliverable** conclusion needs
none of this: the narrow model is better in Rahul's regime by every metric.

### F56 — **`max_freq` CLOSED.** The three combs TIE: it is `max_freq / F`, not `max_freq`.

`exp12`, 4 seeds per arm. Three combs that share a **lowest feature of 126 rad/s** and a
**spacing of 126**, differing only in how far up they run:

| arm | features [rad/s] | `val_th` | `per_window_rms` |
|---|---|---|---|
| F=2, mf=251 | {126, 251} | 8.137e-9 [7.90, 10.7] | 8.480e-5 [8.39, 9.88] |
| F=4, mf=503 | {126, 251, 377, 503} | 8.138e-9 [8.08, 9.14] | 8.563e-5 [8.47, 9.33] |
| F=8, mf=1006 | {126 ... 1006} | 9.017e-9 [8.68, 9.99] | 9.134e-5 [8.80, 9.56] |

**Medians 8.137, 8.138, 9.017 e-9 — they TIE, on every metric, with total overlap.**

That was the pre-registered discriminator: *tie -> the low end; differ -> the top.* **The
top of the comb is irrelevant.** `mf=1006` with F=8 is indistinguishable from `mf=503`
with F=4 — while `mf=1006` with **F=4** (lowest feature 251) was clearly worse. The
parameter was never `max_freq`; it is **`max_freq / F`**, the lowest feature.

**Why the old sweep gave a moving optimum:** it varied `max_freq` at fixed `F=4`, which
moves the top *and* the bottom together. It was never measuring one variable. Sorted by
`max_freq/F` instead, every result in this file lines up — and the "optimum shifts with
W" puzzle (F44) dissolves, because 126, 126 and 188 rad/s at W=20/40/100 is roughly
constant in **absolute** terms while `2*pi/T` spans 251 -> 1257.

**The mechanism, two-sided and consistent with all nine data points:**
- **Too high** and the lowest feature sits above the signal. The deviation is 99.98%
  contained below 126 rad/s (F51), so a comb starting at 251 has nothing to represent.
- **Too low** and it becomes degenerate with the raw `t` the trunk already receives.
  `mf=126, F=4` starts at 31.5 rad/s, which completes **0.06 cycles** across a 12.5 ms
  window — indistinguishable from a straight line. That arm is the worst ever measured.
  126 rad/s completes **0.25 cycles**: the first shape genuinely distinct from linear.

So the lowest feature has to be **high enough to be non-degenerate with `t`, and low
enough to lie inside the signal's band.** At W=20/40/100 that window is 126-188 rad/s.

**What to say:** *"the controlling parameter is `max_freq/F` — the lowest Fourier feature —
which must sit at the signal's bandwidth, ~126 rad/s. The top of the comb is irrelevant:
three combs sharing a lowest feature of 126 tie exactly. Sweeping `max_freq` at fixed F
moved both ends at once, which is why the optimum appeared to depend on window length."*

**F19/F20 are closed**, after being open since Stage B. The remaining honest caveat is
that the two-sided mechanism is a *hypothesis that fits all nine points*, not a proof —
the tie itself, and the rejection of "the top matters", are measurements.

### F55 — **n=10000 buys no measurable improvement. The data saturates near 5000.**

famG = n_runs 10000, dt 50 us, W=40, S=250. 1130 epochs, best at 1070, 8.6 h on MPS.
**Compare it only against famE**, which shares its dt and therefore its noise level — the
famD comparison is confounded by F49, and `val_th`/`per_window_rms` are not comparable
across W, which leaves `rollout_full_rms` and the train/val gap:

| | n_runs | `rollout_full_rms` | train/val gap |
|---|---|---|---|
| famD (dt=100 us, W=40) | 5000 | [3.83, 4.93]e-4 | 1.33 - 1.72 |
| famE (dt= 50 us, W=80) | 5000 | [3.11, 4.54]e-4 | 1.51, 1.57 |
| **famG (dt= 50 us, W=40)** | **10000** | **[2.675, 3.638]e-4** | **[1.38, 1.44]** |

**Doubling the data bought at most a marginal improvement, and nothing significant.**
famG's deployed band **overlaps** famE's (median 3.16e-4 vs 3.83e-4, ~1.2x, n=2 each);
its train/val gap overlaps famD's [1.33, 1.72] though it sits below famE's [1.51, 1.57].

**Be careful how this is stated — the comparison is confounded both ways.** famG shares
famE's `dt` but not its `W`; it shares famD's `W` but not its `dt`. `val_th` and
`per_window_rms` are not comparable across W, and nothing is comparable across dt (F49).
That leaves **only `rollout_full_rms` against famE**, and there the bands overlap. The
per-window and compounding differences (famG 8.87e-5 / ~3.6x vs famE 4.73e-5 / ~8.0x) are
the W=40-vs-W=80 effect, not a data effect.

**The defensible claim: n=5000 -> 10000 does not produce a measurable improvement, at
n=2 seeds and with the comparison confounded.** Not "it does nothing" — it is "we could
not detect anything, at 2x the data and 2x the epochs."

**This closes the extension of F22.** F22 showed 1000 -> 5000 halved the gap (2.60 ->
1.45) and improved `val_th` 1.83x — a real and large effect. 5000 -> 10000 does neither.
**The data saturates around n=5000 for this problem at this architecture**, and the
earlier reading "the gap is still 1.45, so there is headroom" was wrong: a gap of ~1.45
is where this setup sits, not a deficit to be closed with more runs.

**Caveats, and they matter here.** 2 seeds. famG ran 1050-1130 epochs against famE's
~538, so it had roughly double the training budget and still did not separate. Best epochs
were 990 and 1070 of a 1200 cap, so neither was fully converged — which does not weaken
the conclusion, since the extra data *plus* the extra epochs together produced no
detectable gain.

**Consequence for the gains models (exp14):** the worry that a 7-D LHS on 5000 runs would
be too thin is now partly answered — n=5000 is not data-starved at 5-D, and the gap in
those records is the thing to read. If the gains arms come back near 1.45, no more data is
needed there either.

**famG is NOT the flagship.** It is one seed, at a timestep Rahul has not confirmed, whose
apparent edge over famD is the F49 noise artefact. **The shipped model stays
`famD_W40_n5000_W40_F4_mf503_wp0.3_s1sp0.pth`** — median of six seeds, dt=100 us, and the
model every figure is built on. famG's value is as a *measurement*, not a deliverable.

### F54 — **THE `w_phys` OPTIMUM, EXPLAINED.** It trades operator quality against handover.

`exp7` complete, 4 seeds at every weight (`sweeps_famB_wphys`, 32 records, `graphs/14`).
Decomposing deployed error into its two factors — `deployed = per_window x compounding` —
makes the optimum stop being a tuning curve and start being a mechanism:

| `w_phys` | `per_window_rms` (the operator) | `compounding` (the handover) | deployed |
|---|---|---|---|
| 0 | 2.007e-4 | **2.83x** | 5.650e-4 |
| 0.01 | 1.346e-4 | 2.79x | 3.992e-4 |
| 0.03 | 1.080e-4 | 2.72x | 2.992e-4 |
| **0.1** | 9.417e-5 | 2.98x | **2.809e-4** |
| **0.3** | **8.563e-5** | 3.39x | 3.010e-4 |
| 0.6 | 8.914e-5 | 4.08x | 3.634e-4 |
| 1 | 9.430e-5 | 5.25x | 4.860e-4 |
| 3 | 9.090e-5 | **6.08x** | 5.402e-4 |

**The two columns move in OPPOSITE directions.** `per_window_rms` falls 2.34x from
`w_phys=0` to 0.3 and then flattens; `compounding` climbs monotonically 2.83 -> 6.08.
Deployed error is their product, so it has a minimum — at **0.1-0.3**.

**Why the handover degrades, and this is F13/F16 predicting its own failure mode.** The
physics loss is *exactly* gauge invariant under `theta -> theta + a*t + b`,
`omega -> omega + a`: its null space is precisely `(theta0, omega0)`. So it constrains
derivatives and says **nothing** about absolute values. Raising `w_phys` therefore gives
relatively less weight to the **data term — the only thing that pins the absolute
solution** — and the recurrent handover passes *absolute* theta and omega forward. A model
trained with heavy physics is excellent on derivatives and relatively weaker on exactly
the quantity the handover depends on. The gauge argument predicted this before it was
measured.

**The setting.** 0.1 and 0.3 are statistically indistinguishable (`val_th` [8.63, 12.0]e-9
vs [8.08, 9.14]e-9, overlapping). 0.1 has the better median deployed error but a worse
spread ([2.61, 5.60]e-4 vs [2.78, 3.63]e-4 — one bad seed at 5.6e-4). **Keep 0.3**: same
accuracy, tighter. Andreas's instinct to try 0.1 was right and it ties.

**And the confound is closed.** vs `w_phys = 0` at n=5000 with F=4: `val_th` **5.2x**,
`per_window_rms` **2.34x**, deployed **1.88x**, all with no overlap. The Stage A'' banner
comes off: the physics-loss benefit is **not** a data-scarcity artefact, and it is larger
here than the 2.43x measured at n=1000.

**Recurring motif, now seen three times.** Width (F46), residual form (F53) and physics
weight (this) all leave `per_window_rms` nearly untouched while moving `compounding`.
**The one-window operator and the recurrent handover are separately tunable**, and every
knob that looked "neutral" on the operator was actually acting on the handover. Report
both, always; a single deployed number hides which of the two moved.

### F53 — **eq-6 LOSES, and it loses in exactly the predicted way.** `exp10` complete.

`sweeps_famB_eq6` vs the matching `sweeps_famB_wphys` arms, 4 seeds each, `graphs/21`.
eq-6 recomputes `Vq` from the network's own predicted angle, so the physics term **can**
select the solution; eq-4 reads the stored `Vq` and structurally cannot.

| `w_phys` | eq-4 `val_th` [min, max] | eq-6 `val_th` [min, max] | eq6 / eq4 | overlap? |
|---|---|---|---|---|
| 0.03 | [1.00, 1.56]e-8 | [1.32, 1.62]e-8 | 1.09x | yes |
| 0.1 | [8.63, 10.5]e-9 | [9.03, 10.8]e-9 | 1.02x | yes — a tie |
| **0.3** | **[8.08, 9.14]e-9** | [12.7, 15.7]e-9 | **1.69x** | **NO** |
| **1.0** | **[7.85, 10.5]e-9** | [65.6, 89.8]e-9 | **7.91x** | **NO** |

`per_window_rms` says the same: tie, tie, 1.32x, **2.65x**.

**eq-6 is never better, and it degrades MONOTONICALLY with the physics weight.** That is
the signature written down in `exp10`'s header before the run: a self-consistent *wrong*
angle drives the eq-6 residual to zero, so the harder you push on the physics term, the
harder you push toward the spurious minimum. At `w_phys = 0.03` the term barely matters
and the two tie; at `w_phys = 1` the spurious basin dominates and eq-6 is 8x worse.

**Why the ladder was worth 16 jobs.** At the single point `w_phys = 0.3` the answer is
1.69x — real, but dismissible as one setting being tuned and the other not. The **trend**
is what makes it an explanation rather than a number, and it could only be seen by
sweeping the weight.

**What this buys, and it is the best kind of result:** the gauge invariance of F13/F16 is
no longer just a property we noticed and explained — it is **the better design**, and the
alternative was tested and lost. The claim becomes:

> *"The physics loss is derivative supervision whose null space is exactly the initial
> conditions. We also tested the formulation that breaks that invariance — Vq recomputed
> from the predicted angle, which is what inference itself does — and it is never better
> and up to 8x worse, degrading monotonically with the physics weight, consistent with the
> spurious minimum a self-consistent wrong angle creates."*

**Closes** roadmap 0f and open item 6, which has been on the list since the first week.
**Note the incidental cost:** it also means the train/deploy mismatch is *deliberate* —
inference computes `Vq` the eq-6 way because it must (no ground truth at deployment), and
training computes it the eq-4 way because that is better. Say so; do not let it look like
an oversight.

### F52 — **THE ARCHITECTURE ABLATION. The operator beats a plain MLP by ~10x.**

*(Provisional, read from the live logs at 13.3 h — records land when the jobs finish.)*
`exp9`, `sweeps_famB_arch`. Matched dataset, split_seed, seeds, epochs, patience, lr,
batch, `w_phys=0.3`, **and matched F=4 / mf=503** (which required adding `ov` support to
`Single_PINN` — without it the MLP would have silently run at the YAML defaults F=0,
mf=100 and the comparison would have proved nothing).

| | params | `val_th` | s/epoch, same CPU nodes |
|---|---|---|---|
| **Unstacked_DeepONet** | 45 696 | **[7.65, 10.44]e-9** (16 seeds) | **41.4** (median) |
| Single_PINN, h=64 | 29 122 | ~1.005e-6 at epoch 641, still improving | 74.7 (1.8x) |
| Single_PINN, h=96 (capacity-matched) | 46 754 | ~1.253e-6 at epoch 350, still improving | 136.8 (**3.3x**) |

> **CORRECTED once the first records landed. "120x" was an MSE ratio quoted as if it
> were an error ratio — do not repeat that.** `val_th` is a **mean squared** error;
> `per_window_rms` is an RMS. Finished h=64 records (3 of 4 seeds, 800 epochs, 16-17 h):
>
> | metric | Single_PINN h64 | DeepONet (16 seeds) | ratio |
> |---|---|---|---|
> | `val_th` (**MSE**) | [8.30, 12.73]e-7 | [7.65, 10.44]e-9 | **~107x** |
> | `per_window_rms` (**RMS**) | [8.41, 10.42]e-4 | [8.4, 9.6]e-5 | **~10.3x** |
> | deployed RMS | [1.785, 2.139]e-3 | [2.78, 6.43]e-4 | **~5.9x** |
>
> `sqrt(107) = 10.3`, which is exactly the `per_window` ratio — the numbers are
> consistent, the *wording* was not. **Quote "~10x more accurate" (error) or
> "~107x lower MSE" (loss), never "120x more accurate".** A reader hears error.
>
> **Also epoch-capped:** all three ran the full 800 epochs and never early-stopped, so
> the PINN was still improving — the ~10x is an upper bound on the gap. At 10x it is not
> a gap 200 more epochs closes, but say "upper bound" rather than leaving it implicit.
> Same trap as the W=20 arms (see F45's caveat); that is three times now.

**~10x more accurate in error terms (~107x in MSE), at matched capacity.** The h=96 arm exists
precisely so "you starved the MLP of parameters" is unavailable: it has *more* parameters
than the DeepONet and is the worse of the two MLP arms.

**The predicted cost ratio was right.** The a-priori estimate was **3.2x** more
multiply-adds overall for the MLP at W=40 (it re-consumes all 378 voltage samples at each
of 125 query points, while the branch runs once per window). Measured at matched capacity:
**3.3x** slower per epoch on the same nodes. The cost argument no longer needs to be
a-priori — it is measured.

**Now we can say both halves.** Before this, "why an operator?" had only a cost answer and
had to borrow Karampinis's accuracy result on a different component. Now:
*"the factorisation is 3.3x cheaper per epoch AND ~10x more accurate, on our system, at
matched parameters and matched hyperparameters."*

**Note the margin is far larger than theirs.** Karampinis report 38.67e-6 vs 26.87e-6 —
**1.44x** on MSE — for PINN vs Unstacked PI-DeepONet on a synchronous machine. On the
same metric ours is ~107x (~10x in error terms); on MSE-to-MSE that is still ~74x their
gap.
Two honest readings, and the difference should be stated rather than glossed:
the recurrent whole-window formulation may simply demand more of the architecture than a
single-shot trajectory does; or their PINN baseline was tuned and ours runs at the
DeepONet's hyperparameters. **The standing caveat holds: the MLP might prefer a different
`lr` entirely, and we did not search for one.** Say that before someone asks.

**Walltime risk.** At 13.3 h of a 24 h limit, arch[1] (h64) needs ~3 h more and will
finish; arch[8] (h96, 136.8 s/epoch) needs ~17 h more and **will hit the wall** — and a
record is only written on success. If the h96 arms die, read `val_th` out of
`logs/29157171_*.out` directly: the number is in the log even when no JSON is written.
Resource use is healthy either way — 4.1 of 4 cores, 3.4 GB of the 5 GB limit.

### F51 — **`max_freq` RESOLVED.** It was never the top frequency: it is `max_freq / F`.

`src/dft_spectrum.py`, `graphs/20`. The DFT finally answered F19/F20, and the answer is
that the whole sweep was parameterised on the wrong quantity.

**Where the power actually is.** Full-run spectrum of the deviation `theta - (theta0 +
w_base*t)` — the thing the trunk has to represent — 60 validation runs, resolution
12.6 rad/s:

| power below | | |
|---|---|---|
| **17.3 rad/s** (`= sqrt(Ki) = wn`) | **82.7%** | the PLL's own natural frequency |
| 50 rad/s | 99.17% | |
| **126 rad/s** | **99.98%** | |
| 251 rad/s | **100.00%** | |
| 503 rad/s | 100.00% | |

**There is NO power at 503 rad/s.** So `max_freq = 503` was never matching a spectral
feature — F50 already killed the resonance idea, and this kills the bandwidth idea too.

**The mechanism.** `build_trunk_input` places features at `w_k = max_freq * k / F`, so the
**lowest** feature is `max_freq / F` — and *that* is the quantity that has to sit at the
signal's bandwidth. Every result in the file falls out of it:

| config | lowest feature `mf/F` | outcome |
|---|---|---|
| mf=503, F=4 | **126** | **best** — sits exactly at the 99.98% bandwidth |
| mf=628, F=4 | 157 | ties 503 |
| mf=754, F=4 | 189 | worse at W=20/40 |
| mf=1006, F=4 | 251 | worse |
| mf=314, F=4 | 78 | worse |
| mf=100, F=4 | 25 | much worse |
| **F=1**, mf=503 | **503** | **worst** — nothing anywhere near 126 (F50) |
| F=2, mf=503 | 251 | slightly worse |
| F=8, mf=503 | 63 | **ties F=4** — the set still contains 126 |

Every arm that wins has a feature at or near 126 rad/s; every arm that loses does not.
That is nine data points across three window lengths, explained by one number.

**Why the "optimum" appeared to move with W** (F44's 503 / 503-628 / 754 at W=20/40/100):
`max_freq` was being swept at **fixed F=4**, so it moved the top *and* the bottom of the
comb together. The sweep was never measuring what it thought it was.

**PRE-REGISTERED PREDICTION, and it is cheap:** `mf=1006 with F=8` also gives a lowest
feature of 126, so it should perform **like `mf=503, F=4` and unlike `mf=1006, F=4`.** If
it does, the mechanism is established; if it does not, this explanation is wrong. 4 jobs.

> **AMENDED SAME DAY — the causal half of this was wrong.** Andreas asked the obvious
> question: *"does mf=314 not have more features where we care?"* It does, and the answer
> is fatal to the explanation as first written:
>
> | mf (F=4) | features | features in the power band (<=126) | measured, W=20 |
> |---|---|---|---|
> | 126 | 32, 63, 94, 126 | **all four** | **3.94e-3 — WORST** |
> | 251 | 63, 126, 188, 251 | 2 | 3.27e-3 |
> | 314 | 78, 157, 236, 314 | 1 | 3.31e-3 |
> | **503** | 126, 251, 377, 503 | 1 | **2.98e-3 — best** |
> | 1006 | 251, 503, 754, 1006 | 0 | 3.25e-3 |
>
> **Putting MORE features where the power is makes it WORSE.** `mf=126` has all four in
> band and is the worst arm ever tested. So "a feature at the bandwidth" is not the
> mechanism — it merely happens to be true of the winners.
>
> **What survives, and it is still worth a lot:**
> 1. The controlling quantity is **`max_freq / F`**, not `max_freq`. Sorted by it, the
>    W=20 results are unimodal with an optimum at **126-157 rad/s**: 25/32 -> worst,
>    63 -> 3.27, 78 -> 3.31, **126 -> 2.98**, 157 -> ties, 188 -> 3.24, 252 -> 3.25.
>    Sweeping `max_freq` at fixed `F` moved the top *and* the bottom of the comb together,
>    so **the original sweep was never measuring one variable.** That part is solid.
> 2. There is **no power at 503 rad/s**, so no resonance and no matched filter (F50 + the
>    spectrum). Both "physical frequency" stories are dead.
> 3. **Features placed where the power is do not help** — a genuinely surprising fact that
>    points at basis *conditioning* (closely spaced sinusoids are near-collinear over a
>    12.5 ms window, so four of them act like one) rather than at spectral coverage.
>
> **THE DISCRIMINATING TEST, and it is clean.** Three combs that share a lowest feature of
> 126 **and** a spacing of 126, differing only in how far up they run:
> `mf=251 F=2` {126, 251} | `mf=503 F=4` {126...503} | `mf=1006 F=8` {126...1006}.
> **Tie -> it is the low end / spacing. Differ -> it is the top.** 3 arms x 4 seeds.
> Until that runs, F19/F20 are **narrowed, not closed.**

**What to say now:** *"the controlling quantity is `max_freq / F`, with an optimum around
126-157 rad/s, and the original sweep confounded the top of the comb with the bottom. The
signal has no power anywhere near 503, so it is not a resonance — but placing features
where the power IS makes things worse, so the mechanism is about basis conditioning and
is not yet established."*

**Also worth stating on its own:** 82.7% of the deviation's power sits below
`wn = sqrt(Ki) = 17.3 rad/s`. **The signal the operator has to learn is, to first order,
the PLL's own closed-loop response** — which is why the surrogate inherits the loop's
damping (F43) and why the deviation form (D5) works so well.

### F50 — `max_freq` is a BAND, not a frequency. `exp8` complete, 4 seeds per arm.

`sweeps_famB_fcount`, W=40, mf held at **503**, only the number of features varied.
`build_trunk_input` places them at `w_k = max_freq * k / F`, so this is the first
experiment that separates the top frequency from the comb it sits in.

| F | features [rad/s] | `val_th` | `per_window_rms` |
|---|---|---|---|
| 1 | {503} | [1.13, 1.28]e-8 | [9.59, 10.7]e-5 |
| 2 | {251, 503} | [8.28, 10.7]e-9 | [8.73, 9.88]e-5 |
| **4** | {126, 251, 377, 503} | **[8.08, 9.14]e-9** | **[8.47, 9.33]e-5** |
| 8 | {63 ... 503} | [8.08, 9.93]e-9 | [8.42, 9.42]e-5 |

**F=1 loses to F=4 with no overlap on `val_th`.** The single frequency 503 rad/s on its
own is not what matters — so **there is no "resonance at 80 Hz" to find.** F=2, F=4 and
F=8 all overlap: two components are nearly enough and eight add nothing.

**This resolves the F19/F20 mechanism question, in the negative and usefully.** "Why 503?"
is not "what physical line lives at 80 Hz?" — it is **"why is a basis spanning roughly
126-503 rad/s the right one?"** A bandwidth question, not a spectral-peak question. It also
explains the shape of the old `max_freq` sweep without any new hypothesis: at fixed F=4,
raising `max_freq` **coarsens the comb** (spacing = `mf/F`), so `mf=1006` gives
{251, 503, 754, 1006} and abandons the low end — which is why more bandwidth *hurt*. And it
sits comfortably with F44's finding that the optimum is an absolute frequency: a **band** is
an absolute object, while `2*pi/T` is not.

**Consequence for the DFT plan (roadmap 0c):** stop looking for a peak at 500-600 rad/s.
The question worth asking of a spectrum is now **"is the deviation signal's power contained
below ~500 rad/s?"** — a bandwidth check, which is both more likely to succeed and the
thing that would actually explain the number.

### F49 — **F48's dt GAIN IS A NOISE-MODEL ARTEFACT. Retracted 2026-08-20.**

`src/dt_convergence.py`, `graphs/19`. Solver only, no network. Three arms against one
fixed noiseless waveform subsampled onto each grid, noise added afterwards:

| dt [us] | sensors/0.5 s | **noise OFF** | gain | **sigma CONST** (as coded) | gain | **PSD CONST** (physical) | gain |
|---|---|---|---|---|---|---|---|
| 200 | 2500 | 1.792e-6 | — | 1.380e-3 | — | 9.758e-4 | — |
| 100 | 5000 | 4.454e-7 | **4.02x** | 9.406e-4 | 1.47x | 9.406e-4 | 1.04x |
| 50 | 10000 | 1.099e-7 | **4.05x** | 6.851e-4 | 1.37x | 9.688e-4 | 0.97x |
| 25 | 20000 | 2.617e-8 | **4.20x** | 4.710e-4 | 1.45x | 9.420e-4 | 1.03x |

**Three findings, and the second one retracts F48's recommendation.**

**1. The trapezoid is exact for this problem.** Noise off, it converges at a textbook
**4x per halving** and sits at **4.5e-7 rad at 100 us** — *five orders of magnitude*
below anything we measure. So **"discretisation error" has been the wrong name all
along.** Everything in F27/F39/F40/F48 called a discretisation floor is **sensor noise**,
not integration error. The trapezoid was never the limitation.

**2. The 1.58x from halving dt is an artefact of how noise is modelled.**
`_grid_phases` draws `noise_amplitude * (rand - 0.5)` **once per sample, at fixed
amplitude, whatever dt is**. Halving dt therefore halves the in-band noise *power* while
leaving per-sample variance alone — so the error falls by ~sqrt(2) per halving, which is
exactly the 1.47/1.37/1.45 measured. It is the signature of **averaging white noise**,
not of a second-order method. Hold the noise **spectral density** fixed instead — the
physical case, since a real sensor's PSD does not change because you sample it faster,
giving `sigma ~ 1/sqrt(dt)` — and the error is **flat: 1.04x, 0.97x, 1.03x.**

**So: finer sampling buys nothing physical here. F48's "change sensors to 10000 for
accuracy" is WITHDRAWN.**

**3. It also explains why F48 looked so clean.** Both the "floor" and the "network"
components improved by the same 1.58x, and `ours/floor` stayed at 1.096 to three
decimals — because *both* were scaled by the same reduction in noise level. The
underlying claim survives in weakened form: **the surrogate's error is proportional to
the noise level of the target it was trained on, i.e. it adds no independent floor.**
That is still true and still worth saying. What does not follow is "so buy accuracy with
dt".

**THE ACTIONABLE BUG.** `noise_amplitude: 0.1` in `PLL_Constants.yml` is a **per-sample
amplitude**, so the dataset's physical noise content silently changes whenever `sensors`
changes. Any two families at different dt are therefore **not physically comparable**,
famB-vs-famE included. It should be specified as a spectral density and scaled by
`sqrt(dt_ref/dt)` at generation. Until that is fixed, do not compare families across dt
and do not attribute a dt difference to accuracy.

**What this does NOT change:** every within-dt result. W, `max_freq`, `hidden_dim`,
`w_phys`, Fourier features, the head-to-head, the OOD ladder and the lock-in analysis are
all measured at a single dt and are untouched.

**Should we still go to 10000 sensors?** Yes — but for **realism, not accuracy**. Ventura
et al. run their EMT solver at `sim_step = 50 us`, and 50 us is ordinary EMT practice
while 100 us is coarse. Matching their step makes the head-to-head a same-step comparison.
Expect **no accuracy gain** once the noise model is fixed, and expect to **pay 2x compute**
(you emit twice the samples), which turns "tie at half the compute" into "better accuracy
at equal compute". Both are good stories; they are different stories.

### F48 — dt improves the number, and the surrogate sits a CONSTANT 9.6% above the floor.

> **SUPERSEDED BY F49 (same day).** The 1.58x below is real *as measured*, but it is
> caused by the noise model shrinking with dt, not by better integration. Read F49 first.
> The part that survives: the constant 9.6% ratio, reinterpreted.

The test F47 said was needed, run the same morning: both famE checkpoints through
`ood_test.py` against the 12.5 us reference (`logs/ood_famE.txt`, `graphs/18`). Scoring
against true physics instead of against each model's own training solver:

| | total vs 12.5 us | discretisation floor (solver @ own dt) | network alone | **ours / floor** |
|---|---|---|---|---|
| famD, dt = 100 us, W=40 | 1.020e-3 | 9.305e-4 | 4.178e-4 | **1.096** |
| famE, dt = **50 us**, W=80 | **6.446e-4** | 5.882e-4 | 2.637e-4 | **1.096** |

**Halving dt improved the deployed error against true physics by 1.58x** — and the
network component improved by 1.58x too. **The ratio to the floor is identical to three
decimals.**

**This is the strongest single statement in the project.** The surrogate does not add an
independent error floor of its own; it reproduces its training solver to a **constant
relative accuracy**, so its total error is set by the reference it learned from. Practical
consequence: accuracy is bought with dt, not with architecture — and F46 already showed
width does nothing, F22 showed data helps. *(2 seeds per family; the ratios agreeing to
three decimals is partly luck, but the 1.58x is not.)*

**And it retroactively vindicates the F47 caveat.** `rollout_full_rms` showed *no* change
between famB and famE — because it scores each model against its own dt's solve and is
structurally blind to discretisation. The same models, scored against a fixed fine
reference, differ by 1.58x. **Never compare `rollout_full_rms` across datasets with
different dt.** It is a within-dt metric only.

**Bonus finding — shorter windows handle discontinuities far better.** Absolute error on
`jump BIG` (+/-120 deg): famD **~9.0e-3** -> famE **~1.5e-3**, a **6x** improvement where
the control only improved 1.58x. `sag DEEP` improved 1.8x, i.e. only the baseline amount.
So the "+/-120 deg jumps cost 10x" limit in F42 is **largely a window-length artefact, not
a fundamental one** — a step discontinuity inside a 12.5 ms window is harder to represent
than inside a 6.25 ms one. Caveat: famE changes dt *and* W together, so finer sampling and
shorter windows cannot be separated here. Testable with a rewindowed famE at W=40.

**Cost against the ladder:** famE degrades *worse* on `omega_0 x2` (78x vs 36x), as
expected from twice the handovers amplifying an out-of-range branch input.

**THE ORIGINAL CAVEAT THAT LED HERE — this experiment cannot see the thing F39 predicted.**
`rollout_full_rms` is measured against **the dataset's own solve at its own dt**. It
therefore measures fidelity to the training solver, and is *structurally blind* to
discretisation error. F39's ~18% was a claim about error against **true physics**, which
only appears when scored against a finer reference. **So F47 is not a refutation of F39 —
it does not test it.** The test that does: run the famE checkpoint through `ood_test.py`
(control row) or `accuracy_benchmark` against the 12.5 us reference and compare its total
against famB's 1.02e-3. Cheap, and it is the next thing to run.

## WHAT ACTUALLY CHANGES — the decision table, 2026-08-20

The point of the whole sweep programme, in one place. **Four knobs stay, one moves.**

| knob | verdict | why |
|---|---|---|
| `W` (window length) | **keep 40** | F45: W=40/50/100 all inside each other's seed bands. W=10 and W=20 worse by 10.6x and 2.3x. Nothing to gain above 40 |
| `F` / `max_freq` | **keep F=4, mf=503** | F31: Fourier features are worth 4.16x on `val_th`, real at 16 seeds. F44: `2*pi/T`=1257 rejected; 754 wins at W=100 but on 2 seeds and only at W=100 |
| `hidden_dim` | **keep 64** | F46: `val_th` and `per_window_rms` flat from 32 to 128. Width only moves compounding (5.9 -> 3.4), and 128 buys nothing over 64 |
| `w_phys` | **HOLD 0.3, pending** | Stage A'' is confounded (n=1000, F=0). exp7 running |
| **`sensors` / dt** | **no accuracy change — see F49.** Move to 10000 for *realism* (it matches the paper's 50 us EMT step), not for accuracy | F48 measured 1.58x, **F49 retracted it**: the gain was the noise model shrinking with dt, not better integration. Held at fixed noise PSD the error is flat |

**The one-sentence version, corrected 2026-08-20:** *the architecture and every
hyperparameter are saturated — and so is the timestep.* Nothing on this list moves the
number any more. What remains true from F48 is the structural half: **the operator adds no
error floor of its own**, so its accuracy tracks the noise level of whatever it is trained
against. The way to improve it is therefore a cleaner reference, not a finer one.

**Caveats on adopting dt=50 us.** famE is `W=80, S=125` — it halves dt *and* halves the
window. F48's 6x improvement on `jump BIG` is probably the shorter window (discontinuities
resolve better), not the finer sampling, and the two are not separated. It also degrades
worse on `omega_0 x2` (78x vs 36x) because twice the handovers amplify an out-of-range
branch input. And it is 2 seeds. **The clean follow-up is a rewindowed famE at W=40
(S=250)**, which keeps the handover count at 40 and isolates dt from window length —
`src/rewindow.py` cannot do it (it only subdivides), so it needs a fresh generate.

Both famD and famE carry disturbances (`enabled=True, fraction=0.5`), so the fault rows
are like-for-like.

## Figure inventory — what each one is for, and what was removed

| # | shows | source |
|---|---|---|
| 01-06 | dataset + simulator sanity, prediction vs truth, window sweep, residual budget | `pll_plots.py` |
| 07 | `w_phys` sweep **at n=1000, F=0 — CONFOUNDED**, keep only until 14 completes | `sweep.py --plot wphys` |
| 09 | error vs time, and cost vs accuracy, DeepONet band vs solver at two steps | `speed_benchmark.accuracy_benchmark` |
| 10 | Fourier arms (one per family) | `plot_sweeps --kind arms` |
| 11 | W sweep | `plot_sweeps --kind W` |
| 12 | head-to-head vs the paper's NN, inside their trained range | `envelope_figure.py` |
| 14 | `w_phys` at n=5000 with F=4 | `plot_sweeps --kind wphys` |
| 15 | OOD ladder, **absolute** error, all dt families on one axis | `ood_test.py` |
| 16 | the SRF-PLL's own acquisition limit (no network) | `lockin_range.py` |
| 17 | width sweep | `plot_sweeps --kind hd` |

**Removed 2026-08-20:** `08_sweeps_Wtest` and `08_sweeps_ff` (n=1000 Fourier sweeps,
superseded by 10 at 16 seeds on the n=5000 family); `18_ood_famE_dt50` (merged into 15,
which now plots every dt family on one absolute axis — that merge is also what finally
made the 5000-vs-10000-sensor comparison visible in a figure instead of only in a table).

**Still overlapping, not yet merged:** 09's right panel (cost vs accuracy) duplicates 12.
09's left panel (error vs time) is unique. The merge would fold 09-right into 12 and leave
09 as a single-panel figure.

## ROADMAP — agreed 2026-08-19, in order

**Added 2026-08-19 evening, queued ahead of the rest** — both are cheap, both close a
hole that is *currently uncontrolled* rather than merely unexplored:

0a. **`w_phys` at n=5000 with F=4** (`hpc/exp7_wphys.txt`, 28 jobs). The last inherited
    assumption in the project — see the CONFOUNDED banner on Stage A''. 7 values x 4
    seeds; `wp0.3` copied free from `sweeps_famB_ff` (16 seeds). Collect with
    `plot_sweeps.py --kind wphys` (box+strip; `sweep.py --plot wphys` draws min/max
    error bars, which is how F21 went wrong).
0b. **`F` at fixed `max_freq`** (`hpc/exp8_fcount.txt`, 12 jobs). F = 1, 2, 8 at mf=503.
    Separates "top frequency" from "comb spacing", which `max_freq` alone has always
    confounded. The only experiment so far that can move F20's *mechanism* rather than
    its bracket. See "What we can say about `max_freq` TODAY".
0c. **DFT of the residual** — offline, no training, one script. Detailed under the
    `max_freq` section: does anything actually live at 500-600 rad/s?
0e. **Architecture ablation** (`hpc/exp9_arch.txt`, 8 jobs). `Single_PINN` vs the
    operator at matched F/mf/w_phys/seeds, in TWO arms: h=64 (29 122 params, as written)
    and h=96 (46 754, capacity-matched to the DeepONet's 45 696) so "you starved the MLP"
    is unavailable. Closes "why an operator" with our own data. Needed `--arch` in
    `sweep.py` and `ov` support in `Single_PINN` — without the latter it would have
    silently run at the YAML defaults (F=0, mf=100) and compared a Fourier operator
    against a featureless MLP.
0f. **eq-6 residual ladder** (`hpc/exp10_eq6.txt`, 16 jobs). Vq recomputed from the
    predicted angle, at w_phys = {0.03, 0.1, 0.3, 1.0} x 4 seeds — a LADDER paired with
    exp7's, because eq-6 changes the character of the physics term and its optimal weight
    has no reason to match eq-4's. Guarded behind `--residual`, default `eq4`; tags gain
    `_eq6` only when non-default, so no existing tag or record moved. **Verified before
    submitting:** feeding the true theta through `vq_from_prediction` reproduces the
    stored `Vq` to 6.4e-7 relative. Watch for the spurious minimum — a self-consistent
    *wrong* angle also zeroes the eq-6 residual, which eq-4 structurally cannot do.
0d. **OOD ladder** (`src/ood_test.py`) — **queued on the laptop, chained to run when the
    famD seed loop exits** (`logs/ood_test.log`, both famD checkpoints, 32 runs each).
    Evaluation only, no training. Closes "everything so far is interpolation".

### F42 — OOD RESULT (2026-08-20). One hard edge, and it is `omega_0`.

Both famD checkpoints, 32 runs x 0.5 s, vs the 12.5 us reference. `logs/ood_test.log`.

| scenario | ours/control (s0 / s1) | solver/control | ours/solver |
|---|---|---|---|
| in-distribution control | 1.00 / 1.00 | 1.00 | 1.12 |
| freq offset **x5** (+/-1.0 Hz) | 1.05 / 1.11 | 1.06 | 1.11 / 1.17 |
| amp offset **x3** (+/-0.15 pu) | 1.05 / 1.03 | 1.07 | 1.11 / 1.08 |
| sag LONG (0.1-0.3 s, 3x trained) | 1.40 / 1.15 | 1.16 | 1.35 / 1.10 |
| sag DEEP (0.1-0.5 pu, below trained) | **2.60 / 2.15** | 1.16 | 2.51 / 2.06 |
| jump BIG (+/-120 deg, 2x trained) | **9.8 / 8.6** | 1.16 | 9.4 / 8.2 |
| omega_0 **x2** (+/-40 rad/s) | **36.1 / 25.0** | 1.12 | 36.0 / 24.8 |
| omega_0 **x4** (+/-80 rad/s) | **5636 / 5664** | 16.6 | 380 |

**1. Grid-parameter extrapolation is essentially FREE.** 5x the trained frequency offset
and 3x the amplitude offset cost **under 11%** — and the solver degrades by the same 6-7%,
so even that is the scenario being harder to integrate, not the network extrapolating.

**2. Fault severity degrades GRACEFULLY, which is the row that matters.** Sags deeper than
anything trained (0.1-0.5 pu vs 0.5-0.95) cost 2.2-2.6x; sags 3x longer cost 1.2-1.4x.
This is the realistic severe case — a bolted three-phase fault — and the answer is
"degrades, does not break".

**3. Phase jumps past +/-60 deg cost ~9x.** Real, bounded, and worth stating as a limit
rather than discovering in front of an audience.

**4. `omega_0` is the hard edge — but see F43, because most of the cliff is not ours.**
2x the trained range costs 36x, which in absolute terms is 3.6e-2 rad = **2.1 degrees**,
with **0 of 32 runs** off by a whole turn. 4x costs 5.70 rad — but **10 of 32 runs end a
whole 2*pi away**, and wrapping the error collapses 5.70 -> **0.82 rad**. So the
catastrophic number is mostly a DISCRETE cycle-slip miscount, not a loss of precision.
F43 shows the reference loop itself slips 5 cycles there and does not lock until 0.856 s,
i.e. **after the 0.5 s window ends** — the training data contains no settled trajectory
at that `omega_0` to learn from.

**WHY, and this is the part that generalises.** The axes that extrapolate for free enter
only through the **sampled voltage waveform**; the axis that fails is a **direct network
input**. The branch takes `(sin th0, cos th0, (omega0 - mu)/sd)`. A 1 Hz frequency offset
barely changes the waveform across a 12.5 ms window, so the branch still sees an in-family
signal — no extrapolation happens at all. But `omega0` is fed in raw: with `sd ~ 5.3`,
training spans roughly +/-3.8 in normalised units and `omega0 = +/-80` presents **+/-15**.
The first layer is evaluated far outside any input it has seen. **Operator surrogates
extrapolate over their input FUNCTION far better than over their input SCALARS**, and the
fix for the scalars is a dataset change (widen the LHS range on `omega_pll`), not an
architecture change.

**Cross-check that the ladder is calibrated:** the control gives ours 1.012e-3 against a
solver-at-100us floor of 9.06e-4 — we sit **12% above the discretisation floor**, and
`sqrt(clean_deployed^2 + disc^2) = sqrt((3.00e-4)^2 + (8.7e-4)^2) = 9.2e-4` reproduces the
measured 1.01e-3 to within 10%. Consistent with the F39/F40 decomposition, measured a
completely different way.

**The deployable sentence:** *"valid for |omega_0| <= 20 rad/s; within that, grid frequency
and amplitude excursions far beyond the training range are free, and fault severity beyond
the training range degrades gracefully."* Note it costs nothing to say — the trained range
already covers every realistic initial frequency error.

Final numbers, both famD checkpoints, `graphs/15_ood_ladder.png`, `logs/ood_final.txt`:

| scenario | ours s0 | ours s1 | solver@100us | ours/ctrl | turns off |
|---|---|---|---|---|---|
| in-distribution control | 1.020e-3 | 1.013e-3 | 9.305e-4 | 1.00 | 0/32 |
| freq offset x2 | 1.031e-3 | 1.014e-3 | 9.269e-4 | 1.01 | 0/32 |
| freq offset x5 | 1.083e-3 | 1.082e-3 | 9.195e-4 | 1.06 | 0/32 |
| amp offset x3 | 1.066e-3 | 1.059e-3 | 9.318e-4 | 1.05 | 0/32 |
| sag trained | 1.163e-3 | 1.056e-3 | 9.616e-4 | 1.14 | 0/32 |
| sag LONG (3x) | 1.294e-3 | 1.142e-3 | 1.039e-3 | 1.27 | 0/32 |
| jump trained | 1.374e-3 | 1.318e-3 | 9.375e-4 | 1.35 | 0/32 |
| sag DEEP | 2.689e-3 | 2.211e-3 | 1.056e-3 | 2.64 | 0/32 |
| jump BIG (2x) | 1.026e-2 | 8.094e-3 | 9.771e-4 | 10.06 | 0/32 |
| omega_0 x2 | 3.636e-2 | 2.518e-2 | 9.737e-4 | 35.7 | 0/32 |
| omega_0 x4 | 5.704e0 | 5.692e0 | 6.426e-3 | 5595 | **10/32** |

**Reproducibility note (fixed 2026-08-20).** `ood_test.py` now re-seeds immediately before
each `build_case`, so every scenario sees the identical sensor-noise realisation. Before
that, `load_checkpoint` built an MLP — drawing from the global RNG — so **adding a second
checkpoint to the command line silently changed the noise in every row** (jump BIG moved
9.8 -> 10.3 that way). Paired ICs alone were not enough; the forcing has to be paired too.

### F43 — the SRF-PLL's OWN acquisition limit. `src/lockin_range.py`, `graphs/16`.

Asked because of F42: before blaming the network at `omega_0 = 80`, what does the
*reference solver* do there? Pure frequency error, nominal grid, PLL starting
phase-aligned, so `omega_0` is the only variable. Reference solver at 12.5 us, **no
network involved anywhere in this figure**.

`Kp=25, Ki=300` -> `wn = sqrt(Ki) = 17.32 rad/s`, `zeta = Kp/(2*sqrt(Ki)) = 0.722`,
linear 2% settling `4/(zeta*wn) = 0.320 s`.

| omega_0 | Hz error | cycle slips | locked by | |
|---|---|---|---|---|
| 0 - 20 | 0 - 3.2 | **0** | 0.11 - 0.20 s | trained range, settles in under half the window |
| 25 - 40 | 4.0 - 6.4 | **0** | 0.21 - 0.22 s | still clean acquisition |
| 60 | 9.6 | **1** | 0.523 s | **just past the 0.5 s window** |
| 80 | 12.7 | **5** | 0.856 s | 1.7x the window |

**The clean-acquisition boundary is between 40 and 60 rad/s.** Classical theory predicts
the lock-in range at `2*zeta*wn = 25 rad/s`; the measured boundary is ~2x that, so the
textbook estimate is conservative here — quote the measurement, not the formula.

**This explains the F42 cliff, and it largely exonerates the network:**
- `omega_0 = +/-40` is inside clean acquisition, so the surrogate's 36x is a genuine
  extrapolation loss — but it is 2.1 degrees, and no run slips.
- `omega_0 = +/-80` is beyond it. The loop slips **5 cycles** and does not lock until
  **0.856 s**, so within the 0.5 s horizon there is no settled behaviour at all — only a
  slipping transient whose *final turn number* is a discrete, high-sensitivity outcome.
  Asking a surrogate to reproduce which turn you land on is a different problem from
  asking it to track a trajectory. 10 of 32 runs get it wrong; wrapping removes 7/8 of
  the error.

**Note the terminology, because it is easy to state wrongly.** A cycle-slipped run IS
locked — `Vd -> +1`, `Vq -> 0`, it tracks the grid frequency perfectly — it is simply
`k*2pi` off in *absolute unwrapped* angle. Our metric is unwrapped absolute theta, so it
charges 2*pi per slip. That is the right metric for a simulator drop-in (absolute angle
feeds the Park transform downstream), but it should be *said*, not left implicit.

**NOT negative sequence.** An `omega_0` of 80 rad/s is a **12.7 Hz frequency error** in
the PLL's own oscillator (it starts at 62.7 Hz instead of 50). Negative sequence is a
different phenomenon entirely — a grid component rotating at *minus* the fundamental,
which appears at `2*w = 628 rad/s` in the dq frame and which an SRF-PLL genuinely cannot
reject (the reason DSOGI/DDSRF-PLLs exist). Negative sequence is relevant to the
UNBALANCED-fault item in the roadmap; it has nothing to do with the `omega_0` edge.

### The OOD ladder — what it does and why the fault rows matter most

Every number in this project samples initial conditions from one LHS box. `ood_test.py`
pushes past it one axis at a time, with **one shared uniform draw mapped through each
scenario's ranges** so a scenario differs from the control by its range and nothing else.
Three columns per row: our error, the solver at the *same* 100 us step, and their ratio.
That last column is the one that isolates extrapolation — if a scenario is simply harder
to integrate, the solver degrades too and the network is not at fault.

The IC rows (freq offset, amplitude, omega_0 pushed 2-5x) are partly academic: +/-1 Hz off
nominal is a grid with worse problems than its PLL. **The fault rows are not.** We train
on 0.5-0.95 pu sags; a bolted three-phase fault is 0.0-0.3 pu. So the realistic *severe*
case sits outside the training box, which is precisely the case a transient study exists
to examine. `sag DEEP` (0.1-0.5 pu), `sag LONG` (0.1-0.3 s) and `jump BIG` (+/-120 deg)
are the rows to read first.

**Smoke test, and it validated the instrument.** Run at n=2 against the *clean-trained*
n=1000 checkpoint — a model that has never seen a fault — the fault rows blew up
(`sag trained` 5.2x, `jump trained` 11.5x the control) while every IC row stayed within
2x. Exactly the signature expected, which is evidence the ladder measures what it claims.
The real run uses the famD checkpoints, which *were* trained on faults; **the comparison
to look at tomorrow is famD's fault rows against these clean-model numbers** — that
difference is F34's "the surrogate handles faults" stated as a controlled contrast rather
than as a single number.

1. **Disturbances** — sags + phase jumps in `_grid_phases`, done; generate `famD` and
   train the best config on it. **The `roll_rms` in that record will blend clean and
   faulted runs and is comparable to nothing** — the split by `fault_kind` /
   `window_faulted` is required before the number means anything. Build that splitter.
2. **Lean `predict_window`** — skip the two `autograd.grad` calls whose results are
   discarded when `output_dim=2`. Free 3x (58.0 -> 19.4 ms/sim-s). No accuracy cost.
3. **Re-run `head_to_head` and `accuracy_benchmark`** with the best famB checkpoint and
   the lean path — the current numbers used a middling n=1000 model.
4. **W between 40 and 100** (background): W=50 (S=100) and W=125 (S=40) both divide
   N=5000. Cheap at these window sizes.
5. **`max_freq` sanity sweep at W=100**, 1-2 seeds per point — closes the F32 caveat
   above and re-tests F20 at a third window length.
6. **`hidden_dim`** — 64 -> {32, 128}. NOTE it will **not** speed up inference (F25:
   100x the parameters cost the same per sample; the step is overhead-bound). It *may*
   speed up *training*, because SOAP's preconditioner does an eigendecomposition that is
   O(d^3) in the layer width and accounts for ~34% of the step (26.0 ms with SOAP vs
   17.2 ms with Adam at W=40). Accuracy is the real reason to test it: the train/val gap
   has closed to 1.17, so capacity is now the plausible binding constraint, not data.
7. **`sensors` = dt. The largest remaining accuracy lever, ~18%.** Not a nuisance knob:
   `sensors: 5000` with `time_window: 0.5` *is* dt = 100 us. F27 measured the
   discretisation error at 9.25e-4 (100 us) vs 5.78e-4 (50 us) against a network error
   of ~8.7e-4, so total = sqrt(NN^2 + disc^2) goes **1.27e-3 -> 1.04e-3**. Bigger than
   `hidden_dim` or collocation can offer, and it **directly answers F26** — the "2x
   faster per simulated second" objection is entirely the 100 us vs their 50 us step.
   *Clean design, one variable:* `sensors: 10000`, `time_window: 0.5` -> dt = 50 us,
   then **W=80 gives S=125** — identical branch and trunk sizes, 6.25 ms windows, inside
   the F32 flat bottom. Cost: 2x rows, 2x compute, and a new family comparable to
   nothing existing. Sampling margin is not the issue (100 us already gives Nyquist
   5 kHz against the 650 Hz 13th harmonic; D3 is satisfied either way) — the gain is
   the trapezoidal truncation error in the reference being learned.
8. **UNBALANCED faults — and the prediction that would finally explain `max_freq`.**
   famD's sags are **balanced** (one `sag_gain` on all three phases), which is the
   documented prototyping simplification *and* matches the paper exactly — their `CC`
   fault does `ea_array[elem, :] *= step_value`, scaling all three columns. So the
   current comparison is fair; unbalance is a second experiment, not a fix.
   *Why it is not a small change:* unbalance creates negative sequence, which lands at
   **2w = 100 Hz in the dq frame**, and an SRF-PLL structurally cannot reject it (the
   reason DSOGI/DDSRF-PLL exist). For a one-phase drop to 0.5 pu (~0.17 pu negative
   sequence), with the closed loop `H(s) = (Kp*s + Ki)/(s^2 + Kp*s + Ki)`:
   `|H(j628)| ~ 0.040` -> theta ripple ~**6.8e-3 rad**, about **20x the entire current
   deployed error** of 3.0e-4. It would dominate the budget — and it is in the target,
   not a model failure.
   **PREDICTION, pre-registered: `mf = 628 = 2*w_base` should beat 503 once unbalanced
   faults exist.** They tie today (F31, 16 seeds) because nothing lives at 100 Hz. If
   628 wins there, that is the mechanism behind the `max_freq` optimum, open since F19.
9. Frequency-domain validation — lowest value; the sub-10 Hz half needs a longer horizon
   than 0.5 s.

**famD generated and verified 2026-08-19** (`famD_W40`, `famD_W100`, n=5000, disturbances
on): 2500 clean / 1250 sag / 1250 jump, 4.2% of windows flagged at W=40. Sag confirmed to
move the waveform (`|Va|` 1.133 -> 0.763 at depth 0.68, predicted 0.770) — the
`v_nominal + amplitude_offset * sag_gain` precedence bug would have made every sag
invisible, changing the voltage by at most 0.025 pu.

**Operational lesson — the 8 h walltime kill.** The benchmark in `hpc/bench.py` timed
fixed tensors and never touched the `t[idx]` gathers a real epoch performs over a
200000 x 378 branch tensor. Measured effective parallelism on a real job was
**1.9 cores of the 4 requested**, and runs went 500-670 epochs rather than the ~180
Stage C saw on MPS. Result: ~2x the predicted runtime and 34 of 60 jobs killed.
**Never launch a wide array without one full-size calibration job.** `hpc/pending.py`
resubmits only the configs with no record, so a kill costs nothing but time.

## Stage E — benchmarked against the paper's own code (`PINNs-in-EMT`)

Repo cloned into `PLL_Attempt/PINNs-in-EMT`. Three solvers for the same SRF-PLL:
`solver_nr` (Newton-Raphson), `solver_del` (one-step delay), `solver_mlp` (their NN).
All of it reproduced in `speed_benchmark.py`. **Their PLL is our PLL**: `Kp=25`,
`Ki=300`, `f0=50`, and their `res_q = -sin(th)*valpha + cos(th)*vbeta` is exactly our
D1 sign convention. Their state `x1` is our `omega/Ki`. So the equations line up and
the comparison is legitimate.

**The structural difference that governs everything.** Their PLL sits inside a closed
loop — converter + LCL filter + network — so `vq` depends on the network voltage which
depends on `theta_pll`. *That algebraic loop is why they need Newton at all.* Our
`Va,Vb,Vc` is an imposed stiff source, so our Newton only handles trapezoidal
implicitness. **Our problem is easier. Any speed claim has to say so.**

### A. Whole control block — their published metric [ms compute / simulated second]

```
  solver_nr    1084.8
  solver_del    576.7
  solver_mlp    590.5      nr/mlp = 1.84x
```

Their ~60% claim reproduces. **But `mlp` is not faster than `del`** — it is marginally
slower, and `del` is free (no iteration, no network). So their NN does not buy speed
over everything: it buys **`nr`-grade accuracy at `del`-grade cost**, because the delay
trick breaks the algebraic loop with a stale `vq` and pays for it in accuracy. Sharper
framing of their own result than "60% faster" — worth raising with Rahul.

### B. PLL component only — and the finding that kills the size question

| | params | dt | samples/sim-s | ms/sim-s | **us per sample** |
|---|---|---|---|---|---|
| their NN | **450** | 50 us | 20 000 | 39.5 | **1.98** |
| ours, deployed | 45 696 | 100 us | 10 000 | 58.0 | 5.80 |
| ours, lean | 45 696 | 100 us | 10 000 | 19.4 | **1.94** |

**Finding F25 — parameter count is not what sets inference cost here.** 100x the
parameters, *identical* cost per output sample (1.94 vs 1.98 us). Both networks are
dominated by call overhead — their numpy dispatch, our PyTorch dispatch — not by
arithmetic. **Tuning `hidden_dim` for speed is the wrong lever and will not work.**

**Finding F26 — "2x faster per simulated second" was an artefact of the step size.**
We emit one sample per 100 us against their 50 us, so we produce half as many samples
for the same simulated second at the same per-sample cost. Regenerated at 50 us we
would land on their number. The claim is not "faster network", it is **"a window
operator amortises call overhead across 125 steps, so we can take a 2x coarser step"**
— which is only a win if the coarser step does not cost accuracy. Hence C.

**Where speed actually is:** `deployed 58.0 -> lean 19.4`, a **3x** structural win from
not computing two `autograd.grad` calls whose results `predict_window` discards. With
`output_dim=2`, theta and omega are direct network outputs; the derivatives are only
needed for the residual, which inference never uses. **Change `predict_window` before
quoting any speed number.**

### C. Accuracy — the test that actually decides it

`accuracy_benchmark()` in `speed_benchmark.py`. One grid-voltage realisation on a
12.5 us grid; coarser grids **subsample that same waveform** rather than regenerating
it (`_grid_phases` draws fresh `torch.rand` noise per sample, so a regenerated 50 us
run would see a different physical signal and the comparison would be void).
24 runs x 0.5 s, theta RMS against the 12.5 us trapezoidal solution:

| | theta RMS [rad] | max | cost [ms/sim-s] |
|---|---|---|---|
| solver @50 us | **5.777e-4** | 2.207e-3 | 2179.2 |
| solver @100 us | 9.251e-4 | 3.345e-3 | 1174.0 |
| DeepONet @100 us, 4 ckpts | **1.122e-3 .. 1.743e-3** | 5.8-8.8e-3 | **~61** |

Per checkpoint, next to the `rollout_full_rms` already on record for each:

| checkpoint | accuracy RMS | ratio vs solver@100us | recorded roll_rms |
|---|---|---|---|
| n=5000 seed 1 | **1.122e-3** | **1.21x** | 5.751e-4 |
| n=1000 seed 0 | 1.273e-3 | 1.38x | 7.149e-4 |
| n=5000 seed 0 | 1.399e-3 | 1.51x | 9.550e-4 |
| n=1000 seed 1 | 1.743e-3 | 1.88x | 1.126e-3 |

**The two columns rank-order identically.** This benchmark draws 24 *fresh* initial
conditions (`torch.manual_seed(0)` in `build_case`, same ranges as the YAML, not the
dataset's validation split), so that is an independent confirmation that
`rollout_full_rms` measures what we think it measures. Best cross-check in the file.

**Finding F27 — the surrogate error and the discretisation error are the same size.**
Treating them as independent, the network's own contribution at the median checkpoint
is `sqrt(1.273^2 - 0.925^2) = 8.7e-4` rad — which lands on that checkpoint's recorded
`rollout_full_rms` of 7.1e-4. Two independent measurements agreeing to ~20%.

**Checkpoint choice is worth 1.55x** — as much as most of the effects being chased in
the sweeps. **Never quote this benchmark from one checkpoint** (`accuracy_benchmark`
now takes a list and prints the band). And note **1.00x is the ceiling, not a target**:
the network is trained to reproduce `solver@100us`, so it cannot be more accurate than
its own teacher. The n=5000 pair *brackets* the n=1000 pair rather than beating it,
which is F24 again — `roll_rms`-like metrics stay seed-noisy even at n=5000, while
`val_th` is where the n=5000 advantage actually shows.

Consequences, and they are the actionable part:
- **1.38x worse than the solver at the same step, for 19x less compute.** That is the
  honest headline. Against the solver at *their* step it is 2.20x worse for 35x less.
- **Neither term can be improved alone.** Halve the step and discretisation drops to
  5.8e-4 while the network's 8.7e-4 dominates. Perfect the network and you stop at the
  9.3e-4 discretisation floor. They are balanced, so a real accuracy gain needs
  **both** a better network *and* a finer step.
- This is the quantitative reason `hidden_dim` is not the lever (F25 says the same
  thing about speed, from the other direction).

Plot: `graphs/09_accuracy_vs_step.png` — left, error vs time for the three; right,
error vs compute, "down and left is better".

**Method note, and it is the n_eval=20 lesson for the third time.** At `n_runs=4` this
benchmark reported DeepONet 9.404e-4 vs solver@100us 9.458e-4 — i.e. "the surrogate is
free". At `n_runs=24` it is 1.273e-3, **35% worse**. The four-run result was a
small-sample fluke of exactly the kind that produced F21 and the `n_eval=20` retraction.
**Do not quote this benchmark below ~24 runs.**

**Caveat on the 50 us row.** Each grid sees a decimated version of the fine noise
sequence, so the 50 us and 100 us solvers integrate *different* realisations of the
sensor noise. That is physically what a coarser sensor does, but it means the
50-vs-100 gap is not a pure discretisation measurement (observed ratio 1.60, where
clean 2nd-order convergence would give 4). **The DeepONet-vs-solver@100us comparison is
unaffected** — both see the identical decimated waveform.

### D. HEAD TO HEAD against their NN — it IS possible, and it is the best result yet

Their network is a **self-contained PLL stepper**, not something welded into their
network solve:
```
in:  (x1_n, theta_n, vq_n, valpha_{n+1}, vbeta_{n+1})   out: (x1_{n+1}, theta_{n+1})
```
— exactly the inputs their Newton solve receives. So it can be driven by *our* grid
voltage with no converter/LCL model at all. `paper_nn_at()` does this. Mapping:
their `x1` = our `omega/Ki`; their `vq = -sin(th)*valpha + cos(th)*vbeta` expands
algebraically to our Park q-row, same D1 sign. They wrap theta to `[0, 2pi]` each step.

**Their trained envelope, read off the scaler in `version1.npz`:**

| input | theirs | ours |
|---|---|---|
| `x1` (= omega/Ki) | +/-0.050 (omega +/-15) | +/-0.087 |
| `vq` | **+/-0.30** | **+/-1.14** |
| `valpha`, `vbeta` | +/-1.10 | +/-1.16 |
| `theta` | `[0, 2pi]` wrapped | unwrapped |

**Their NN is built for a PLL that is always near lock** — that is why `vq` never
leaves +/-0.3. Ours starts up to +/-pi/2 out of phase and drives `vq` to +/-1.14.
Comparing on our full envelope measures *their extrapolation*, not their accuracy, so
`head_to_head(near_lock=True)` restricts to the intersection of the two envelopes
(eps0 within +/-0.05pi, omega0 within +/-12).

theta RMS vs a 12.5 us reference, 12 runs x 0.5 s:

| | NEAR-LOCK (fair) | FULL envelope | cost [ms/sim-s] |
|---|---|---|---|
| solver @100 us | 9.747e-4 | 9.781e-4 | ~1060 |
| **their NN @50 us** | **1.038e-3** | 4.810e-3 | 62.4 |
| their NN @100 us | 8.087e-3 | 9.450e-3 | 30.9 |
| **ours @100 us** | **1.038e-3** | 1.464e-3 | 61.7 |

**Finding F28 — in the regime both were designed for, the two surrogates are
equivalent.** 1.038e-3 vs 1.038e-3, at 62.4 vs 61.7 ms/sim-s. Same accuracy, same
cost. Neither architecture wins. That is worth saying plainly and it is a fair result
for a 450-parameter step model against a 45 696-parameter window operator.

**Finding F29 — ours holds accuracy at a 2x coarser step; theirs does not.** Forced to
100 us their NN degrades **7.8x** (1.038e-3 -> 8.087e-3) while ours is *defined* at
100 us. Their update is explicit Euler on a learned derivative, `x + dt*NN(x)`, so the
truncation error is tied to the step it was trained at. A window operator carries no
such tie. **This is the coarse-step claim from F26, finally demonstrated instead of
asserted** — and it is the one architectural advantage that survives scrutiny.

**Finding F30 — ours degrades gracefully outside its envelope, theirs does not.**
Full envelope vs near-lock: theirs 4.6x worse (1.038e-3 -> 4.810e-3), ours 1.4x
(1.038e-3 -> 1.464e-3). Consistent with the `vq` range table — theirs is extrapolating
~4x beyond training. Our 5-D LHS over the full +/-pi/2 acquisition range buys exactly
this. **Their design target is a converter already synchronised; ours includes
acquisition.** Different jobs, and the harder one partly explains our accuracy gap in C.

**THE CAVEAT THAT LIMITS ALL OF D — and it is a big one.** These runs contain **no
voltage sags and no phase jumps**. `build_case` calls `_grid_phases`, which produces a
steady sinusoid plus harmonics plus noise; nothing happens mid-run. But their NN *was*
trained with faults in its envelope (their `sim_setups` use `CC` 0.8 pu for 25 ms and
`PJ` 12.5 deg) and our model has **never seen one**. So D is run entirely on OUR
scenario type, which favours us. A symmetric comparison needs a common test containing
disturbances, and **we would probably fail it today**. This upgrades HANDOVER item 8
from "paper gap" to "the missing half of the only head-to-head we have".

*Bug worth remembering:* the first run of D gave their NN `RMS 4.443 rad`, which looked
like total failure. It is `sqrt(0.5 * (2pi)^2)` — half the runs sitting exactly one turn
off, because their theta is wrapped to `[0, 2pi]` and ours is absolute. Unwrapping
restores continuity but not the branch; re-anchoring to the true start fixed it.
**A "the other method is broken" result that is exactly 2pi is never the other method.**

### F41 — "56x cheaper than the solver" is a BATCH-SIZE-1 statement. Say so.

Found 2026-08-19 evening while checking whether the cost axis in figs 09/12 is
apples-to-apples. **It is not, and the reason is a normalisation asymmetry in the code:**

- `deeponet_at` and `paper_nn_at` divide elapsed time by `n_runs` -> **per trajectory**.
- `solve_at` divides by `horizon` only, **not** by `n_runs` -> **per batch**.

So the solver point on both figures is the cost of simulating all 24 (fig 09) or 32
(fig 12) trajectories, plotted against two per-trajectory points. The number survives by
luck rather than by design: the solver is fully vectorised in torch, so its wall time is
nearly **flat** in batch size, and the batch figure therefore happens to sit close to the
true single-trajectory cost.

Measured on the laptop, same dt = 100 us, same 0.5 s horizon, same process
(`time_solver` vs `time_surrogate(lean=True)`), ms of compute per simulated second **per
trajectory**:

| batch | trapezoidal solver | DeepONet (lean) | ratio |
|---|---|---|---|
| 1 | 1021.5 | 19.22 | **53x** |
| 32 | 33.80 | 3.54 | 9.5x |
| 128 | 9.59 | 2.18 | 4.4x |
| 512 | 3.78 | 1.60 | **2.4x** |

**What this means.**
1. **The 53-56x is real and it is the right number for the EMT use case** — a simulator
   integrates *one* trajectory, and at batch 1 the vectorised solver has nothing to
   amortise over while its Python loop over 5000 steps pays in full.
2. **The advantage collapses under batching, from 53x to 2.4x.** The solver batches
   almost for free (512 trajectories cost 1.7x one trajectory), whereas our rollout is a
   recurrent Python loop over windows and amortises less well. We still win at every
   batch size measured, but "56x" is indefensible without the batch size attached.
3. **This is exactly the axis Karampinis et al. claim on** — their >30x single-trajectory
   grows to 6720x at 1000 trajectories, because they batch the network on GPU and the
   RK45 baseline runs sequentially on one CPU core. Our solver baseline is *already*
   vectorised, which is why our curve moves the opposite way. Worth stating explicitly:
   the two speedup claims are not comparable, and ours is the more conservative baseline.

**Quote it as:** "53x faster than the trapezoidal solver for a single trajectory,
narrowing to 2.4x at batch 512, against a fully vectorised baseline."

**TODO (small, and it removes the asymmetry):** make `solve_at` divide by `n_runs` like
the other two, and re-plot 09/12. The plotted solver point moves left by the batch size,
which is the honest picture: *at the batch size the figure actually uses*, the gap is ~10x,
not ~56x. Both figures then need the batch size in the caption.

### On removing the Python/numpy overhead

F25 says both networks are overhead-bound, so the obvious next thought is to rewrite
inference in a compiled language. **They already did it**:
`PINNs-in-EMT/PSCAD_FORTRAN_SETUP/NN_eval_for_PSCAD.f90`, 117 lines, the 450 weights
unrolled as `#LOCAL REAL` arrays for a PSCAD custom component. That is their answer to
the deployment-cost question and it is worth reading before writing anything in C++.
A compiled version of ours would measure the *arithmetic* cost — genuinely interesting,
since F25 says we are nowhere near it — but it would not change C, and C is what decides
the science.

**Bug in their repo, worth reporting.** All three solvers do
`assert sim_step <= sim_plot_step` in `init_sim_saver` on a **bare name** — a module
global that exists only when the file runs as `__main__`. None of the three can be
imported as shipped; `speed_benchmark.load_paper_solver` injects it. Third instance of
this class of bug across two codebases (see "Bugs found in the draft").
Also: running `python solver_nr.py` directly calls `save_arrays("sim_1")`, which
**overwrites the paper's own shipped reference trajectories** in
`Saved_trajectories/Sim_4events/`. Never run their scripts directly.

---

## HANDOVER — what is left in the middle

> **SUPERSEDED 2026-08-20 — DO NOT WORK FROM THIS LIST.** Every "do first" item below is
> finished: the n=5000 re-baseline, the famB family, exp1 (the Fourier comparison, F31)
> and exp2 (the W sweep, F32/F45) all landed. It is kept because the *reasoning* about
> seed counts and memory scaling is still correct and still useful when sizing a new
> array. **The live lists are: START HERE (open questions), WHAT ACTUALLY CHANGES (the
> decision table), and ROADMAP (items 0a-0f).**

**Do first, in this order:**
1. ~~Regenerate at `n_runs=5000` and re-baseline~~ **DONE** — `pll_dataset_n5000_W40.npz`
   exists (1.0 GB, n_runs=5000 confirmed in its meta) and Stage C is the re-baseline.
   Still to do: the multi-W family `famB_W{10,20,40,100}` at n=5000 for Stage D
   (`hpc/job_generate.sh`). Measured: the ODE solve is ~7 s; wall time is
   `savez_compressed`; peak RSS ~8-9 GB. Run it alone.
2. **Stage D exp1 — the W=40 Fourier comparison at n=5000**, F=0 vs mf=503, **8 seeds
   each**. Not 2-3: F24 says `roll_rms` keeps a 1.66 seed spread even at n=5000, and on
   an array 8 seeds costs the same wall-clock as 2. F21 is withdrawn, so this is the
   first clean measurement of this comparison at any n.
3. **Stage D exp2 — the W sweep**, never done under the fixed loss. `famB` family,
   `rollout_full_rms` at n = W, 3 seeds. Memory scales as `batch_size * S`: W=40 ~2 GB,
   W=20 ~3 GB, **W=10 ~5 GB** for the autograd graph, plus ~3 GB resident data at
   n=5000. Keep batch_size=512; on LSF ask for 16 GB rather than dropping parallelism.

**Open but lower priority:**
4. `mf=628` (`2*w_base`) — the last untested mechanism candidate for the 503 optimum.
5. **Collocation points** — the one remaining PINN idea with headroom (F11). Requires
   `interp1d` on `Va,Vb,Vc` at random times; `pll_infer.interp1d` already does the hard
   part.
6. **`hidden_dim`** — capacity has never been touched. Only worth it after (1); F22 says
   capacity is not currently the binding constraint.
7. **Seed-variance study at W=40** (F=0 vs mf=503, 6 seeds each) — subsumed by (2) if
   that resolves it.

**Paper gaps (TPWRS2026, Rahul is a co-author) — worth more to the supervisor than
another 5% on roll_rms:**
8. **No disturbances in the dataset.** The paper tests voltage sags (0.965->0.8 pu for
   25 ms) and 20 deg phase jumps. Our 1000 runs vary only the *initial condition* —
   nothing happens mid-run. A PLL surrogate that has never seen a fault is not a
   transient-study tool. Modest change to `_grid_phases`. **Biggest scientific gap.**
9. ~~**No speed benchmark.**~~ **DONE — Stage E.** Benchmarked against the paper's own
   `PINNs-in-EMT` repo, not just against our own solver. Headline: **1.38x worse than
   the trapezoidal solver at the same 100 us step, for 19x less compute** (F27), and
   **the same cost per output sample as their 450-parameter network** (F25). New work
   this opened up:
   - **Make `predict_window` skip the discarded autograd** — a free 3x (58.0 -> 19.4
     ms/sim-s). Do this before quoting any speed number.
   - **Re-run C at 50 us** to test F26 directly: does a 50 us dataset put us on their
     per-simulated-second number, with the discretisation error halved?
   - A compiled inference path, if the arithmetic cost is ever worth measuring — but
     read their FORTRAN component first.
10. **No frequency-domain validation** (their Fig 12: 800 points, 1 Hz–1 kHz Bode of
    surrogate vs equation-based model). Stronger than any time-domain RMS.
11. **Input domain is ~20x narrower than theirs** (amplitude offset +/-0.05 pu vs their
    `v_abc` in [-1.2, 1.2] V_nom; our `dt` is fixed where their NN takes `dt` as input).

**Scope note to raise with Rahul:** their `NN^PLL` predicts ONE `dt` step and slots into
the EMT co-simulation loop. Ours predicts a whole window and needs `Va,Vb,Vc` for that
window up front — which in closed-loop EMT is not available, since the terminal voltage
depends on the PLL output. **Ours is a standalone/replay emulator, not a drop-in
co-simulation component.** Name this before a reviewer does; ask him which framing he
wants.

**Housekeeping left over:**
- ~~`sweeps_ndata/` holds 2 stale duplicates~~ **DONE** — already deleted; `sweeps_ndata/`
  holds exactly the 4 comparable records.
- **Plot freshness (checked 2026-08-18).** `08_sweeps_Wtest.png` postdates its
  re-evaluation and is current. `07_sweeps_wphys.png` predated it and has been
  regenerated. `08_sweeps_ff.png` predates its re-evaluation **and cannot be correctly
  regenerated** while 2 of its 10 records are void — leave it stale rather than redraw
  it from bad records, and delete it once Stage D supersedes the figure.
- `runs/W40_F4_mf503_wp0.3_s{0,1}sp0.pth` are new-family checkpoints under old-family
  names (F21 banner). Seed 0 is byte-identical to
  `runs/pll_dataset_W40_n1000_...s0sp0.pth`; seed 1 is a separate earlier 651-epoch run.
  **Not deleted** — decide before cleaning `runs/`.
- `dataset_generator.py`'s `__main__` block is commented out (it used to overwrite
  `pll_dataset.npz` on any accidental run).
- `main()` now records `n_eval_runs`; records written before that show `null`. All of
  those were re-evaluated at 150 via `reval.py`.

**Tooling built this session:** `reval.py` recomputes rollout metrics for *existing*
checkpoints at any `n_eval_runs` (~5–10 s each) — use it instead of retraining whenever
the evaluation protocol changes.

**HPC: DONE — see Stage D and `hpc/README.md`.** `docs/HPC workshop.pptx` is the LSF
runbook (login3.hpc.dtu.dk, `module`, `bsub`/`bstat`/`bkill`, `#BSUB` directives).
`docs/HPC Usage.xlsx` is a resource survey, not instructions — it says the group uses
the `elektro` queue and the central HPC, nothing operational. Queues: `hpc` (CPU),
`gpuv100`/`gpua100`/`gpul40s` (GPU, 24 h cap). Home 30 GB and backed up; `/work3/$USER`
large and not. The headline is in Stage D: **the cluster is 4x slower per run and the
GPUs cannot be used properly on a 45k-parameter model — the win is running 33 jobs at
once, which is what finally makes an 8-seed comparison affordable.**

**What is NOT affected by this retraction:**
- **The W=20 hypothesis rejections stand.** They are within-W=20 comparisons with clean
  separation at n_eval=150: 503 beats 251 (`2*pi/T`) and 314 (`w_base`) with no overlap
  and spreads of 1.02–1.08. F19/F20 hold.
- **The physics sweep is STRENGTHENED.** At n_eval=150, spreads fell to 1.03–1.11 and
  `w_phys=0.3` [1.446, 1.571] beats `0` [3.442, 3.538] by **2.31x with no overlap** —
  and also beats 0.03 [2.214, 2.284], 0.1 [2.159, 2.367] and 1 [1.628, 1.804], all with
  no overlap. Stage A'' is confirmed at higher precision.

**Finding F20 — the optimum is a fixed ABSOLUTE frequency, not a fixed
cycles-per-window.**
```
W=40  ->  optimum 503 rad/s      = 1.00 cycles/window
W=20  ->  optimum 503-754 rad/s  = 2.00-3.00 cycles/window
```
- **basis (2*pi/T): REJECTED.** Predicted 251 at W=20; val_th says no, no overlap.
- **w_base = 314: REJECTED.** Tested directly at W=20, clearly worse than 503.

The parameter-free-rule hope is dead. Something fixed near 500-750 rad/s is wanted.

**Leading untested candidate: `2*w_base = 628.3 rad/s`**, sitting in the middle of the
winning band and *physically motivated rather than fitted*: the per-phase sensor noise is
independent, which creates negative-sequence content at the fundamental, which lands at
**2w in the dq frame**. So `Vq` genuinely carries 100 Hz content, reaching theta through
the `Kp` path. **Post-hoc — do not claim until tested. Test: `mf=628` at W=20 AND W=40.**

**NOTE — the n_runs test did NOT happen.** `pll_dataset_n5000_W40.npz` was generated
with `n_runs=1000` (the YAML edit did not take; the 201 MB filesize was the giveaway —
5000 runs is ~1 GB). Separately, the run tag did not include the dataset, so baseline and
test wrote to the *same filename*. Both fixed: tag now carries the dataset stem and
`n_runs`. The two surviving records are a valid new-family W=40 baseline
(roll 7.43e-4 / 1.25e-3), moved to `sweeps_W40_newfamily/`. **Redo the n_runs test.**

**Level note (needs its own check):** W=20 rollout ~5.0e-3 vs W=40's 1.1e-3, and
per-window 2.1e-3 vs 3.0e-4 — a 7x worse operator, offset only partly by half the
handovers (comp 2.3 vs 4.0). Suggests W=40 really does beat W=20, but the W=20 numbers
are on the NEW LHS family and W=40's are on the old one, so this is not yet a valid
comparison. The W sweep must be run entirely within the new family.

## Presentation — the defence sheet

Rewritten 2026-08-20. Everything above this line is a chronological lab log; **this
section is the claims-first version**, and it is what slides get built from. Each claim
carries its number, its figure, its caveat, and what would falsify it. If a claim is not
in this table, do not put it on a slide.

### The one sentence

> A DeepONet surrogate of an SRF-PLL, applied **recurrently** over 0.5 s with 40 handovers
> and no ground truth, holds the grid angle to **0.017 degrees** — matching the solver it
> learned from to within **1.05-1.11x at 53x less compute**, and matching the published
> step-surrogate's accuracy at **half the compute** — through voltage sags and phase jumps.

### The claims — number, figure, caveat, falsifier

| # | claim | number | figure | caveat | what would falsify it |
|---|---|---|---|---|---|
| 1 | Deployed accuracy over 40 recurrent handovers, clean runs | **3.00e-4 rad = 0.0172 deg** | 03, 04 | 6 seeds, [2.58, 3.20]e-4 | a seed outside that band |
| 2 | Error growth is **sub-diffusive**, and saturates | **2.9x** over 40 handovers vs **6.3x** = sqrt(40) for an undamped random walk | 04 | ratio of deployed to teacher-forced, so "later windows are easier" divides out | growth at or above sqrt(N), or still climbing at N=40 |
| 3 | Matches its own solver at the same timestep | **1.05-1.11x** the solver's error, **53x** cheaper at batch 1 | 09, 12 | **always say the batch size** — it is 2.4x at batch 512 (F41) | a batched comparison quoted as if unbatched |
| 4 | Ties the published NN inside its trained range, at half the compute | tie (0.6%), **2.0x** cheaper per simulated second | 12 | restricted to their trained `vq` range on purpose | their network doing better inside its own envelope |
| 5 | Our **network alone** is more accurate than theirs | **3.4x** (1.84e-4 vs 6.33e-4) | 12 | error decomposition assumes the two sources are independent | a finer reference collapsing the gap |
| 6 | Their step model cannot take a coarser step; ours can | theirs degrades **9.3x** at 100 us; ours is defined there | 12 | this is the one *architectural* claim | a step model holding accuracy at 2x its trained dt |
| 7 | Disturbances cost accuracy but do not break it | sag **1.61x**, jump **2.00x** vs clean | 15 | 6 seeds; the per-seed ratio bands do **not** overlap | overlapping ratio bands at more seeds |
| 8 | The surrogate adds **no error floor of its own** | **9.6%** above the noise-driven floor — *the same 9.6% at half the timestep* | 15 | 2 seeds per family; and both terms moved together because both are noise-dominated (F49) | the ratio moving with dt |
| 9 | Every knob is saturated — W, width, `max_freq` **and** dt | all flat. dt's apparent 1.58x was the noise model shrinking (F49) | 11, 17, 19 | figure 19 is solver-only, so this one needs no network caveat | a knob that still moves the number at fixed noise PSD |
| 10 | The physics loss is **derivative supervision**, provably | null space is exactly `(theta0, omega0)` — three lines of algebra | 14 | true of the **eq-4** form we chose, not of PI losses in general | nothing: it is a proof, not a measurement |
| 11 | ... and it is worth a large factor, not a marginal one | **5.8x** `val_th`, 2.5x operator, 1.9x deployed | 14 | at n=5000 **with** Fourier features — the confound is closed | — |
| 12 | Fourier features are real | **4.16x** `val_th`, no overlap at **16 seeds** | 10 | `max_freq` optimum is empirical | — |
| 13 | Extrapolates over the input **function**, not over input **scalars** | freq at **5x** trained: 1.06x. amp at **3x**: 1.05x. `omega_0` at 2x: **36x** | 15 | | any scalar input extrapolating freely |
| 14 | The `omega_0` cliff is the **loop's** limit, not the network's | beyond ~40 rad/s the PLL itself cycle-slips and locks only at 0.523-0.856 s | 16 | the network still fails there — 0.82 rad after forgiving slips | the loop acquiring cleanly where the network fails |

### Figure by figure — what each one is for

| # | shows | the line to say |
|---|---|---|
| **01** | 5-D LHS coverage | "the input space is stratified, not gridded — four of five marginals flat; `theta_pll` is ragged **by design** because it is tied to the grid angle" |
| **02** | the simulator locks | "validated against `atan2(Vbeta, Valpha)`, a reference that never touches the loop. Settling ~0.3 s matches `wn = 17.3`, `zeta = 0.72`" |
| **03** | prediction vs truth, incl. a sag and a jump | "indistinguishable. Errors start at zero and decay — the PLL damps injected state error" |
| **04** | error vs number of handovers | **claim 2.** "rises to n~5, then falls. 2.9x at n=40 against 6.3x for a random walk" |
| **05** | R^2 per window | "teacher-forced, so this measures the operator, not the deployed system. ~1.00 everywhere" |
| **06** | residual budget | "a *perfect* model would still score the dropped `Kp*dVq/dt` term — the residual floor is set by the formulation, not by fit quality" |
| **09** | error vs time | "where the error lives in the 0.5 s — not concentrated at the handovers" |
| **10** | Fourier arms, 16 seeds | **claim 12.** "every seed shown. The bar for a real difference is non-overlapping min/max on `val_th`" |
| **11** | W sweep | **claim 9.** "flat from W=40 to 100; only 10 and 20 are worse" |
| **12** | accuracy vs cost, vs the paper's own code | **claims 3-6.** "their code, run directly — not reimplemented. Restricted to the range their released model was trained on" |
| **14** | `w_phys` sweep at n=5000 with F=4 | **claim 11.** "and this is the sweep that closed the confound: the old one was at n=1000 with no Fourier features" |
| **15** | OOD ladder, absolute error, both timesteps | **claims 7, 8, 13.** "light bars are the discretisation floor at each timestep. Purple sits left of blue on every row" |
| **16** | the loop's own acquisition limit | **claim 14.** "no network anywhere in this figure — this is the reference solver alone" |
| **17** | width sweep | **claim 9.** "flat from 32 to 128 on the operator; only the *handover* changes" |
| **19** | does a finer timestep buy anything? | **claim 9.** "solver only. Integration error is five orders below the noise, and the apparent dt gain is the noise model shrinking — at fixed noise spectral density the error is flat" |

### The five questions you will be asked, and the answers

1. **"You used 100x their parameters."** True — 45,696 vs 450. And **the cost per output
   sample is identical**: 1.94 vs 1.98 us, because both are dominated by call overhead,
   not arithmetic (F25). So the extra capacity is free *on the axis that decides
   deployment*. Concede the parameter count first, then point at the cost column. The
   claim is not "our bigger model wins", it is **"at equal cost per sample, they left
   3.4x of accuracy unclaimed"**.
2. **"3100x better than baseline?"** That baseline is an open-loop clock that never reads
   the input. It is the null hypothesis, not a method. **Lead with the solver and their
   NN.** The trivial baseline earns one line — it proves the network is doing PLL work
   rather than fitting the 50 Hz ramp, which is a real failure mode when ~94% of the
   signal *is* that ramp.
3. **"Why `max_freq` = 503?"** Empirical, and honestly so. `2*pi/T` and `w_base` are
   **both rejected** at two window lengths with no overlap; the optimum sits at the same
   absolute frequency across a 5x change in window length, which is what a window-relative
   explanation forbids. *(Do not invent a mechanism. "Two natural hypotheses tested and
   rejected, optimum bracketed" is a publishable statement on its own.)*
4. **"Does it generalise?"** Measured, not asserted — figure 15. Grid frequency at 5x and
   amplitude at 3x the trained range cost under 11%. Faults **deeper and longer than
   trained** degrade gracefully. One hard edge, `omega_0`, and figure 16 shows it is where
   **the loop itself** stops acquiring inside the window.
5. **"Only 0.5 s?"** The operator is window-local — it maps one window to one window and
   has no notion of total horizon, so a longer rollout needs no retraining. What 0.5 s
   limits is the *reference* available to score against, not the method.

### Honest limitations — say these before you are asked

- **Training budget.** 5000 trajectories against their 450-parameter model. The cost table
  excuses the parameter gap; it does not excuse this one.
- **Balanced faults only.** One `sag_gain` on all three phases. This matches the paper
  exactly, so the comparison is fair — but unbalanced faults create negative sequence at
  `2w = 628 rad/s` in dq, which an SRF-PLL structurally cannot reject. That is a second
  experiment, not a fix.
- **`|omega_0| <= 20 rad/s`.** Beyond ~40 the loop cycle-slips. Costs nothing to say: the
  trained range already covers every realistic initial frequency error.
- **Not yet a drop-in EMT component.** A standalone/replay emulator; co-simulation
  coupling is not demonstrated.
- **`max_freq` has no mechanism**, only a bracket and two rejections.
- **Some arms are 2 seeds** (`hidden_dim`, `mf` at W=100, famE). Where a claim rests on 2
  seeds it says so, because n=2 is exactly what produced the retracted F35.

### Two explainers to have ready

**Why 6.3x is the right thing to compare compounding against.** Each handover injects a
small error `e`. Two extremes bracket what can happen over N = 40 of them:

| model of the handover error | growth after N=40 | why |
|---|---|---|
| **coherent drift** — every handover pushes the same way | **40x** | the errors add linearly |
| **random walk** — independent, random signs | **6.3x** = sqrt(40) | the *drunkard's walk*: a drunk taking N random steps ends up sqrt(N) steps from the lamppost, not N, because the steps partly cancel. This is the default expectation for **any** recurrent scheme with no damping |
| **measured** | **2.9x** clean, 3.2-3.5x with faults | **below even the undamped case** |

Landing below the random walk is the claim: the handover errors are not merely
uncorrelated, they are actively **damped**. The surrogate inherited the PLL's own
closed-loop stability (`zeta = 0.72`), so an injected state error decays instead of
persisting to be added to by the next window. Two causes are mixed in the *fall* over the
second half of figure 04 — the loop's damping, and the physics simply getting easier once
the PLL locks at ~0.32 s — which is exactly why `compounding` is defined as a **ratio to
the teacher-forced error over the same windows**: the second cause divides out.

**Which baseline to lead with.** Three exist; only two are comparisons.

| baseline | what it is | result |
|---|---|---|
| **trapezoidal solver @ same dt** | the thing the network was trained to reproduce, so **1.00x is a ceiling, not a target** | 1.05-1.11x its error, 53x cheaper at batch 1 |
| **the paper's own NN** | a published surrogate for the same component, run from their code | tie inside its trained range, half the compute |
| `theta0 + w_base*t` | an open-loop clock: never reads the input, cannot lock, cannot see a sag | 9.44e-1 rad — **sanity floor only** |

### Numbers never to quote

| do not say | say instead |
|---|---|
| ~~4.0e-3 rad / 236x~~ | 3.00e-4 rad, and lead with the solver comparison |
| ~~compounding 4.0~~ | **2.9** on clean runs (the 4.0 is the n=1000 model) |
| ~~"56x cheaper"~~ unqualified | **53x at batch 1**; 2.4x at batch 512 |
| ~~"3.3x cheaper than their NN"~~ | **2.0x** per simulated second — the 3.3x included our own Python driver around their network |
| ~~"the error contracts"~~ | "grows 2.9x over 40 handovers, below the 6.3x of a random walk, and saturates" |
| ~~"5.3x more accurate on the full envelope"~~ | only inside their trained range. Outside it, their model is extrapolating and the comparison is not a comparison |
| ~~`graphs/13`~~ | deleted. Figure 12 is the head-to-head |
| ~~`gauge_check.py`~~ | deleted. Cite the three-line proof instead |
| ~~"halving dt buys 1.58x accuracy"~~ | **retracted (F49)** — that was the noise model shrinking with dt. At fixed noise PSD the error is flat. Still true: `rollout_full_rms` is blind to the noise floor and must never be compared across timesteps |


## To check / experiment with next — CLOSED OUT 2026-08-20

This was the working list from the early days. Every item is now resolved; kept as a
record of what was predicted to matter versus what actually did.

| # | item | outcome |
|---|---|---|
| 1 | `w_phys` sweep at `{0 .. 1e-6}` | **DONE, and the range was wrong by 5 orders.** The optimum is 0.1-0.6, not 1e-7. Confirmed at n=5000 with F=4 (5.8x on `val_th`) |
| 2 | actually run SOAP; make `lr` reach the optimiser | **DONE** — both in `main` |
| 3 | train longer (run A never early-stopped) | **DONE** — `epochs=800, patience=40`; nothing hits the cap |
| 4 | long-run dataset for Stage 7 | **DROPPED, and correctly.** The operator is window-local, so a longer horizon needs no retraining and no new family — only a longer *reference* to score against. Cost was overestimated by a day |
| 5 | bounds guard in `rollout` | **DONE** — raises if `n_windows > W` |
| 6 | eq-4 vs eq-6 residual | **RUNNING** (`exp10`), behind `--residual`, default unchanged. Turned out to be the sharpest open question in the project — it decides whether the physics loss *can* select the solution |
| 7 | baselines: `Single_PINN`, trivial ramp | **`Single_PINN` RUNNING** (`exp9`) — the last hole in Stage 8. Trivial ramp done, and demoted to a sanity floor |
| 8 | 3 seeds x best config | **DONE, 6 seeds** on famD, plus 16 on the famB reference arm |
| 9 | shorter window (0.5 s / 5 windows) | **SUPERSEDED** by the full W sweep: W=40/50/100 tied, W=10 and W=20 worse (F45) |

## Stage status — updated 2026-08-20

```
Stage 0  environment + baseline            DONE
Stage 1  physics core                      DONE
Stage 2  reference simulator               DONE   settled Vd -> +1, Vq -> 0, omega
                                                  correlation 0.99992, window joins exact
Stage 3  dataset builder                   DONE   LHS verified; disturbances added (F34)
Stage 4  the operator                      DONE   two heads (F10) -- the step change
Stage 5  physics layer (autograd)          DONE   and PROVEN gauge-invariant on paper,
                                                  not just numerically (see START HERE)
Stage 6  training loop                     DONE   divergence guard, atomic records,
                                                  arch/residual flags
Stage 7  recurrent rollout                 DONE   3.00e-4 rad clean, growth 2.9x over 40
                                                  handovers vs 6.3x for a random walk
Stage 8  baselines, seeds, results         DONE   6 seeds on famD; benchmarked against
                                                  the paper's OWN code, not a reimpl.
                                                  Only gap: Single_PINN (arch, running)
Stage 9  hyperparameter programme          DONE   W, max_freq, hidden_dim, w_phys,
                                                  sensors. Verdict: only dt moves the
                                                  number -- see "WHAT ACTUALLY CHANGES"
Stage 10 robustness + envelope             DONE   OOD ladder (F42), lock-in range (F43)
Stage 11 write-up                          <- HERE. notes -> defence sheet -> slides
```

**What "done" does not mean.** Five arrays are still on the cluster (`w_phys`, `fcount`,
`arch`, `eq6`, `famE_W40`); the DFT is unwritten; the final model has not been retrained
at 10000 sensors. None of those block the write-up — they refine numbers already in it,
except `arch`, which fills the one acknowledged hole in Stage 8.
