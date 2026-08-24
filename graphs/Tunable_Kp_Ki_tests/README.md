# PLL DeepONet — models for the EMT co-simulation

Eight models, all **dt = 100 us** (5000 samples per 0.5 s), all trained with voltage sags
and phase jumps in the data. Every one is a *whole-window operator* applied **recurrently**:
give it the window's `Va, Vb, Vc` plus the state you handed it, and it returns `theta` and
`omega` across that entire window.

## Pick one

|  | window | calls / simulated second | ms / sim-s | theta @ Kp=25, Ki=300 |
|---|---|---|---|---|
| **`famD_W40 …s1sp0`** | 12.5 ms | 80 | 19.4 | **2.39e-4 rad**  ← best accuracy |
| **`famI_W20 …s0sp0`** | 25 ms | **40** | **10.9** | 4.03e-4  ← half the compute |
| **`famJ_W40 …s0sp0_g`** | 12.5 ms | 80 | 20.6 | 5.85e-4  ← **Kp, Ki tunable** |
| `famJ_W20 …s0sp0_g` | 25 ms | 40 | 11.2 | 1.07e-3  ← tunable + cheap |
| `famH_*`, `famK_*` | | | | narrow-`omega` variants — no advantage, see below |

All scored the same way: 12 fresh trajectories, `omega_0` in +/-2, `Kp=25 Ki=300`, full
0.5 s recurrent rollout with no ground truth fed back.

**narrow vs wide** is the range of initial PLL frequency error the model was trained on:
narrow `omega_0` in +/-2 rad/s, wide +/-20 rad/s (includes cold acquisition).

> **Use the WIDE model.** We built the narrow ones expecting them to win in your regime and
> they do not: scored on one common test set, wide and narrow **overlap** at both window
> lengths, and wide's median is lower at W=40 even in a +/-0.2 rad/s band. An earlier
> "narrow is 1.4x better" reading was an artefact of scoring each model on its own
> validation split, and those splits are not equally hard. So there is no specialist to
> choose — the wide model is at least as good where you run it *and* survives cold
> acquisition. `famD_W40 …s1sp0` (fixed) or `famJ_W40 …s0sp0_g` (tunable gains).

**fixed vs GAINS** (`_g` suffix): the `_g` models take `Kp` and `Ki` as **inputs**, so you
can retune the PLL without retraining. It costs ~2.5x on angle error at your tuning and
~5% on inference time. A fixed-gain model is **not** approximately right at a nearby
tuning — at Ki=200 instead of 300 it is 38x worse. See `03_gain_sensitivity.png`.

## Calling it

```python
from pll_infer import predict_window
from train_pll import load_checkpoint

model, ck = load_checkpoint("famH_W40_n5000_W40_F4_mf503_wp0.3_s0sp0.pth")
S = ck["data_meta"]["S"]              # samples per window: 125 at W=40, 250 at W=20
t = ck["t_local"]                     # window-local time grid
t_ext = torch.cat([t, t[-1:] + (t[1] - t[0])])   # one extra point = the handover state

theta, omega = predict_window(model, ck, theta0, omega0, Va, Vb, Vc, t_ext)
#  _g models additionally take kp=..., ki=...  and REFUSE to run without them
#  Va/Vb/Vc are the S samples of that window; theta0/omega0 are scalars
#  theta[:-1], omega[:-1] are the window; theta[-1], omega[-1] are what you feed forward
```

`theta` is **absolute and unwrapped**, ready for the Park transform. Gains models need
`kp=` and `ki=` — they raise rather than silently assuming 25/300.

## Files

`*.pth` are self-describing: architecture, normalisation and the dataset `meta` are stored
inside, so `load_checkpoint` rebuilds them without any config file. The `.py` files and
`config/` here are copies of what they were trained with, for reference.

## Known limits

- **`|omega_0| <= 20 rad/s`** (wide) or **`<= 2`** (narrow). Past ~40 rad/s the *loop
  itself* cycle-slips and does not lock inside 0.5 s, so there is nothing to learn.
- Grid frequency and amplitude excursions well beyond the training range cost <11%.
- Faults deeper and longer than trained degrade gracefully (~2.6x at 0.1 pu sags).
- Gains models: stay inside roughly **Kp 18-45, Ki 180-520**, and ideally Kp 25-41,
  Ki 200-500. Your tuning (25/300) sits comfortably inside that.

  It is **not** the underdamped corner that hurts, which is what you would guess. The
  damage follows *low `Kp`* and *low `Ki`* independently of `zeta` — see
  `03_gain_sensitivity.png`:

  | | worst cell | its `zeta` |
  |---|---|---|
  | `Kp=10` column | 3.6e-3 to 8.7e-3 at every `Ki` | 0.20 to **0.50** |
  | `Ki=100` row | 1.0e-3 to 6.7e-3 at every `Kp` | 0.50 to **2.50** |
  | sweet spot `Kp` 25-41, `Ki` 200-500 | 6.2e-4 to 1.2e-3 | 0.56 to 1.45 |

  `Kp=10, Ki=100` is `zeta = 0.50` — properly damped — and still 6.7e-3, the second-worst
  cell in the box. `Kp=50, Ki=100` is `zeta = 2.50`, heavily *over*damped, and still
  1.8e-3. So a damping-ratio rule would send you to the wrong place: it would keep the
  `Ki=100` row, which is the worst row we measured, and discard the good high-`Ki` region.

  One caveat on reading that figure: the colour is an **absolute** angle RMS. At `Ki=100`
  the loop's natural period is 0.63 s, longer than the 0.5 s window, so those runs are
  still in transient at the end and `|theta|` is simply larger. Part of the edge effect
  may be "the answer is bigger here" rather than "the model is relatively worse here".
  Either way the practical advice is the same, and your operating point is interior.
- Balanced faults only — no negative sequence in the training data.
