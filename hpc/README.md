# Running the sweep on DTU HPC

Everything here is new infrastructure. No file outside `hpc/` is touched — `sweep.py`,
`train_pll.py` and the configs run unmodified on the cluster.

Source material: `docs/HPC workshop.pptx` (login, modules, LSF batch scripts) plus the
DTU HPC docs for [batch jobs](https://www.hpc.dtu.dk/?page_id=1416),
[GPU queues](https://www.hpc.dtu.dk/?page_id=2759) and
[disk quota](https://www.hpc.dtu.dk/?page_id=927).

---

## 0. Get the code there

```
ssh <user_id>@login3.hpc.dtu.dk
```

From the laptop, push the code but **not** the datasets or checkpoints (`--exclude`
matters: `pll_dataset*.npz` is 2 GB today and the datasets are regenerated on the
cluster anyway):

```
rsync -avz --exclude '.venv' --exclude '*.npz' --exclude 'runs' --exclude '__pycache__' \
      ~/Desktop/DTU/Summer_Internship_2026/PLL_Attempt/ \
      <user_id>@login2.hpc.dtu.dk:~/PLL_Attempt/
```

**`login2.hpc.dtu.dk`, not `transfer.gbar.dtu.dk`.** The transfer host is sftp-only and
has no remote `rsync`, so it fails with `protocol error: unexpected tag 94` — which
looks like corruption and is really "that binary does not exist there". Add `-n` for a
dry run first. No `--delete`, so nothing on the cluster is removed.

### Driving it from the laptop without logging in

`ssh host 'cmd'` runs a **non-interactive** shell, which does not source the profile
that puts LSF on `PATH`. Every `bsub`, `bjobs`, `bstat` and `getquota_*` invocation
therefore dies with `bash: line 1: bsub: command not found`. Wrap them:

```
ssh <user_id>@login2.hpc.dtu.dk 'bash -lc "cd ~/PLL_Attempt && bsub < hpc/job_gen_nofault.sh"'
ssh <user_id>@login2.hpc.dtu.dk 'bash -lc "cd ~/PLL_Attempt && sh hpc/submit.sh hpc/exp16_nofault_gains.txt nofault"'
ssh <user_id>@login2.hpc.dtu.dk 'bash -lc "bjobs -w"'
```

Plain commands (`grep`, `ls`, `tail`) need no wrapper. Note also that `bjobs -A` drops
**completed** array elements from the summary — an array that shrinks has finished, it
has not been killed.

Home is 30 GB and backed up; `/work3/<user_id>` is large and **not** backed up. One
`famB` family at `n_runs=5000` is ~4 GB (4 files × ~1 GB), which fits in home. Check
with `getquota_zhome.sh` / `getquota_work3.sh`. If home gets tight, put the `.npz`
files on `/work3` and symlink them into the project root — the code opens them by
relative path, so a symlink is transparent.

## 1. Build the environment (login node, once)

```
sh hpc/setup_env.sh          # CPU torch, for the hpc queue
sh hpc/setup_env.sh gpu      # CUDA torch, for gpuv100/gpua100
```

`requirements.txt` is pinned for the laptop's python 3.14 — `numpy==2.5.2` needs python
>=3.12 and the newest cluster module may be 3.11. The script handles this: it loads the
newest python it can find, tries the exact pins, and falls back to the same packages at
the latest version that python accepts. It then writes `hpc/env_lock.txt` and runs
`hpc/smoke_test.py`, which does one real optimiser step and prints the step time.

A relaxed environment is fine — all 60 jobs share this one `.venv`, so cluster runs stay
mutually comparable. It is one more reason not to compare them against laptop runs.
(In practice module `python3/3.13.0` resolves the exact pins, so the cluster ends up on
the same package versions as the laptop and only the *device* differs.)

**Disk.** On Linux the default torch wheel drags in ~2.5 GB of NVIDIA CUDA libraries that
the `hpc` CPU queue never touches, and pip caches another ~2.5 GB. After setup:

```
pip cache purge          # reclaims ~2.5 GB
getquota_zhome.sh        # 30 GB total; venv ~3 GB + famB family ~4 GB
```

Do **not** re-run `setup_env.sh` once jobs are running — it does `rm -rf .venv` and would
swap the environment underneath them.

## 2. Generate the dataset family

`config/initial_conditions.yml` already has `n_runs: 5000`, so nothing to edit.

```
export STEM=famB
bsub < hpc/job_generate.sh
```

Writes `famB_W10.npz famB_W20.npz famB_W40.npz famB_W100.npz` — one LHS draw, one ODE
solve, four windowings, which is what makes the W sweep vary only W.

> **The one genuinely destructive step in this whole pipeline.** `generate_multi_W`'s
> default `path_fmt` is `pll_dataset_W{W}.npz` — the existing n=1000 family that all 12
> `sweeps_Wtest/` records point at. `LatinHypercube` is called with no seed, so an
> overwritten dataset is gone for good and every record naming it becomes
> un-revaluable. `hpc/generate_family.py` refuses to overwrite unless you pass
> `--force`; use it rather than calling `generate_multi_W` directly. The same trap
> catches `pll_dataset_n5000_W{W}.npz`, which would collide with the Stage C dataset.

Cost, measured on the laptop: the ODE solve is 1.37 ms/step × 5000 steps ≈ 7 s. Wall
time is almost entirely `np.savez_compressed` on ~1 GB per file. Peak RSS ~5.5 GB
during `_grid_phases` alone, ~8–9 GB with windowing and save, hence 16 GB reserved.
Run it alone — it is one job, not an array.

## 3. Submit a sweep

```
sh hpc/submit.sh hpc/exp1_w40_fourier.txt  ffw40     # 48 jobs: F=0 | mf=503 | mf=628, 16 seeds each
sh hpc/submit.sh hpc/exp2_w_sweep.txt      wsweep    # 12 jobs, W = 10 / 20 / 100, 4 seeds
```

Both can be in the queue at once — they are independent and write to different
`results_dir`s.

`submit.sh` strips comments, prints the numbered job list, and derives the `[1-N]`
array range from the same expanded file that `sed` indexes — so the range and the
line numbers cannot drift apart. Read the printed list before it goes.

```
bstat                 # or bjobs -A for the array summary
bkill -J ffw40        # kill the whole array
tail -f logs/<jobid>_<index>.out
```

For the GPU queue, pass the GPU script as a third argument:

```
sh hpc/submit.sh hpc/exp1_w40_fourier.txt ffw40 hpc/job_sweep_gpu.sh
```

## 4. Collect

```
python sweep.py --collect --results_dir sweeps_famB_ff --plot ff
python sweep.py --collect --results_dir sweeps_famB_W          # table only
```

`exp2` deliberately omits W=40 because `exp1` already trains it. Once `exp1` is done:

```
cp sweeps_famB_ff/famB_W40_*F4_mf503* sweeps_famB_W/
```

Copying the record is enough — the `ckpt` path inside stays valid.

---

## Which queue

Measured on the laptop, batch 512, F=4, ms per optimiser step:

| W | S | MPS (M1 Max) | CPU 1 thread | CPU 4 | CPU 8 |
|---|---|---|---|---|---|
| 100 | 50 | 7.5 | 48.2 | 38.9 | 38.6 |
| 40 | 125 | 17.2 | 107.2 | 83.5 | 77.1 |
| 20 | 250 | 31.2 | 183.3 | 117.9 | 119.5 |
| 10 | 500 | 61.1 | 342.2 | 195.9 | 189.7 |

(The 17.2 ms at W=40 reproduces the 8.1 s/epoch in the Stage C record, so the numbers
are calibrated against a real run.)

So **a CPU job is ~4× slower per run than the laptop**, and 4 threads is the knee —
8 threads bought under 10%.

**The win is width, not speed.** Total data volume is fixed, so every W costs roughly
the same per run (~25–30 min on MPS, ~1.5–2 h on a cluster CPU node). The 60 jobs in
exp1 + exp2 are ~28 h serial on the laptop with the laptop unusable; as one array they
are one longest-job wall-clock, ~2 h, and the laptop stays free.

And that is what makes the seed problem tractable. F24 says `roll_rms` keeps a 1.66 seed
spread even at n=5000 with subsampling noise removed, so the only cure is more seeds —
which on a laptop was "you'd need 40 seeds, not feasible" and on an array is free.
**Spend the cluster on seeds, not on hoping for a faster GPU.** exp1 is 16 seeds per arm
for exactly this reason.

**On the GPUs.** DTU has V100/A100/L40s nodes and they are genuinely powerful, but this
model cannot use them. It is 45k parameters; one optimiser step is roughly 4 GFLOPs
against a V100's ~15 TFLOP/s, so the card would be idle ~98% of the step waiting on
kernel launches. A GPU helps when the model is big or the batch is huge — here
`batch_size` is pinned at 512 for comparability, so neither applies. On top of that the
rollout evaluation at the end of `main()` runs on CPU regardless (`load_checkpoint`
defaults to `device="cpu"`), batch size 1, ~12k calls at W=40/`n_eval=150` — pure GPU
idle time. Expect maybe 1.5–2× over MPS, not 10×, and pay for it in queue wait.

Submit one config to each queue and compare `seconds` in the two JSON records if you
want the real number. For a 60-wide array, `hpc` will almost certainly win on
wall-clock simply because it has far more slots.

## Things that will bite

- **Do not mix devices inside one comparison.** `batches()` shuffles with
  `torch.randperm(n, device=device)`, so CPU, MPS and CUDA draw different minibatch
  orders from the same seed, and float reductions differ too. A cluster run and a
  laptop run at seed 0 are different runs. Against a 1.6× seed spread, a device
  confound is not survivable — that is why `exp1` re-runs *both* arms rather than
  pairing new `F=0` runs against the existing MPS `mf=503` records.
- **`--n_eval_runs 150` is not the default.** `sweep.py` defaults to 20. Every line in
  the config files passes it explicitly; keep it that way. Note that at `n_runs=5000`
  the validation set is 750 runs, so 150 is no longer "the whole validation set" as it
  was at n=1000 — it is the same standard error as before, not zero subsampling noise.
- **`--split_seed 0` on every line**, varying only `--seed` (F16).
- **One `results_dir` per family.** The tag now carries the dataset stem, so
  checkpoints no longer collide — but keeping families in separate directories means
  `--collect` and `reval.py` cannot silently mix them.
- **`reval.py` re-scores existing checkpoints**; never retrain because the evaluation
  protocol changed. It reads `r["ckpt"]` as a relative path, so run it from the project
  root. It rewrites the JSON in place — back the directory up first.
- Walltime on GPU queues is capped at 24 h; `-W 8:00` here is already generous for the
  longest run (W=100, ~2 h).
