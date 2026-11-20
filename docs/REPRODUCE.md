# Reproducing the numbers

Everything in the paper comes from one scene (`data/fieldH`) and two training entry points. This page is the shortest path from a fresh clone to the tables in `results/`.

## 0. Environment

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`gsplat` compiles its CUDA kernels the first time a rasterisation call happens. That needs:

- `nvcc` (CUDA toolkit matching your PyTorch build) — on Linux usually already present;
- `ninja` (`pip install ninja`) — the Python package puts the binary on PATH for `torch.utils.cpp_extension`;
- a CUDA-capable GPU; `TORCH_CUDA_ARCH_LIST` set to your architecture if auto-detection fails (e.g. `export TORCH_CUDA_ARCH_LIST="12.0"` for Blackwell);
- Windows only: the MSVC C++ build tools, run from a `vcvars64.bat` shell (PyTorch shells out to `where cl`).

Verify:

```bash
python -c "import torch, gsplat; print(torch.__version__, torch.cuda.get_device_name(0))"
```

## 1. Data sanity check (no GPU needed)

```bash
python src/verify_data.py --manifest     # counts + held-out split + hashes, both scenes
python src/selftest.py --data_dir data/fieldH --factor 8 --max_images 60
```

`verify_data.py` expects `fieldH` 331 images / 329 poses / 137,229 points / 41 held-out and `jxfx` 117 / 117 / 78,192 / 14, and prints `VERIFY_OK`; it also regenerates `data/MANIFEST.md`, whose hashes let anyone confirm the download matches what the tables were computed from.

Expected: `视角 60，初始点 137229`, all images get an altitude from `altitude.csv`, `1 归一化单位 = 99.73 m`, 329 poses, `ROI 14 px@22 m → 6 px@118 m`, `SELFTEST_OK`. This exercises the loader, the altitude fallback, the H localisation and the ROI projection without touching the rasteriser.

## 2. Train

Protocol of the paper: `factor 4`, 30,000 iterations, densification window 12,000, SH degree 2, cap 3 M Gaussians, held-out views `range(7, n, 8)` (41 views), three seeds.

```bash
DATA=data/fieldH
ROOT=outputs/v4_fieldH
COMMON="--data_dir $DATA --factor 4 --max_images 331 --max_steps 30000 \
        --sh_degree 2 --refine_stop 12000 --max_gaussians 3000000"

for s in 0 1 2; do
  # standard baseline (w = 1)
  python src/mini_train.py $COMMON --seed $s --out $ROOT/seed${s}_base

  # TPS, w = 3 (ROI weighting only, densification criterion untouched)
  python src/train_ours.py $COMMON --seed $s \
      --loss_mode roi --h_roi_weight 3 --h_radius_m 6 --salience_strength 0 \
      --out $ROOT/seed${s}_ours

  # weight sweep
  python src/train_ours.py $COMMON --seed $s \
      --loss_mode roi --h_roi_weight 5 --salience_strength 0 --out $ROOT/seed${s}_w5

  # Local Densification-Only baseline (w = 1 + densification salience)
  python src/train_ours.py $COMMON --seed $s \
      --loss_mode roi --h_roi_weight 1 --salience_strength 3 --salience_radius_m 8 \
      --out $ROOT/seed${s}_salonly

  # controls
  python src/train_ours.py $COMMON --seed $s --loss_mode focal     --out $ROOT/seed${s}_focal
  python src/train_ours.py $COMMON --seed $s --loss_mode randroi   --out $ROOT/seed${s}_randroi
  python src/train_ours.py $COMMON --seed $s --loss_mode randfix   --randfix_offset_m 25 --out $ROOT/seed${s}_randfix
  python src/train_ours.py $COMMON --seed $s --loss_mode gradscale --h_roi_weight 3 --out $ROOT/seed${s}_gradscale
done
```

One 30k-iteration run at `factor 4` takes ≈23 min and ≈2.0 GB peak memory on an 8 GB RTX 5070 Laptop, ending at ≈1.99 M Gaussians. The full grid is 24 runs — trim the seeds to `0` for a first pass.

Each output folder contains `mini_trained.ply`, `raw_cam_pos.npy` and `metrics.txt`.

## 3. Evaluate

```bash
python src/eval_recon_v4.py --data_dir data/fieldH --factor 4 --root outputs/v4_fieldH --seeds 0 1 2
```

It loads the scene once, renders the 41 held-out views for every `seed<k>_<config>/mini_trained.ply`, and reports full-image PSNR/SSIM plus marking-region H-PSNR/H-SSIM/sharpness (evaluation region = projected square of 4 m object-space radius). Results are written next to `results/` as `v4_recon_metrics<tag>.csv`; aggregation into the mean ± SD table is `python src/aggregate_v4.py`.

Expected (33 views in which the marking appears, three seeds):

| config | full PSNR | H-PSNR | ΔH-PSNR |
|---|---|---|---|
| base | 24.24 ± 0.15 | 22.01 ± 0.25 | — |
| ours (TPS w = 3) | 24.27 ± 0.04 | 22.74 ± 0.20 | +0.74 [+0.58, +0.90] |
| salonly | 24.15 ± 0.15 | 21.76 ± 0.08 | −0.25 |

Small deviations (<0.1 dB) are normal: the rasteriser, cuBLAS and the adaptive densification decisions all contain non-deterministic kernels even with a fixed seed. The paired comparison (same seed, same views) is what the paper reports.

## 4. Mechanism probes (optional, single seed)

```bash
# criterion-gradient probe (Table 1) and accumulated-gradient trace (Table 2)
python src/train_ours.py $COMMON --seed 0 --loss_mode roi --h_roi_weight 3 \
    --salience_strength 0 --grad_hook --out outputs/probe_w3_clean
python src/train_ours.py $COMMON --seed 0 --loss_mode roi --h_roi_weight 3 \
    --salience_strength 0 --event_log --out outputs/probe_w3_events

# synthetic viewpoints (for the closed-loop / detector side, not used in the paper)
python src/render_synthetic_v2.py --data_dir data/fieldH --ply outputs/v4_fieldH/seed0_ours/mini_trained.ply \
    --tag ours --out outputs/synthetic
python src/eval_synth_closed.py --help
```

`grad_hook` writes `grad_trace.csv` (per-view marking-region vs. background gradient magnitudes, normalised by τ); `event_log` writes `event_trace.csv` with clone/split counts inside the 1.5 / 2 / 3 / 4 / 6 m bands. Neither changes training behaviour.

## 5. Consecutive-radial-band and stress evaluation

`results/` already holds these runs; to regenerate them you need the same checkpoints:

```bash
python src/eval_altitude.py --data_dir data/fieldH --ply outputs/v4_fieldH/seed0_ours/mini_trained.ply \
    --factor 4 --out outputs/eval_ours_altsweep
```

`eval_altitude.py` sweeps the evaluation radius and bins by altitude, which is where the 0–4 / 4–6 / 6–10 / 10–14 m table and the altitude-tier MAE figures come from.

## 6. Second scene (Table 10)

`data/jxfx` is the campus scene: 117 images, `sparse/0`, `altitude.csv`. Train the two configs the table compares (baseline and loss-only, three seeds):

```bash
COMMON="--data_dir data/jxfx --factor 4 --max_images 117 --max_steps 30000 \
        --sh_degree 2 --refine_stop 12000 --max_gaussians 3000000"
for s in 0 1 2; do
  python src/mini_train.py $COMMON --seed $s --out outputs/jxfx/seed${s}_base
  python src/train_ours.py $COMMON --seed $s --loss_mode roi --h_roi_weight 1 \
      --salience_strength 3 --salience_radius_m 8 --out outputs/jxfx/seed${s}_lossonly
done
python src/analysis/eval_jxfx_multi.py       # writes results/jxfx_multi_metrics.csv
```

The evaluation uses `range(7, 117, 8)` = 14 held-out views, matching the 14 rows per config in the shipped CSV.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `gsplat: No CUDA toolkit found. gsplat will be disabled` | `nvcc` not on PATH | add the CUDA `bin` directory to PATH, restart the shell |
| `Ninja is required to load C++ extensions` | missing ninja | `pip install ninja` |
| `'_C' object has no attribute 'CameraModelType'` / `AttributeError: NoneType` | kernels not built | run once inside the CUDA/MSVC environment; delete `~/.cache/torch_extensions` if a half-built cache exists |
| `Unknown CUDA arch () or GPU not supported` | arch auto-detection failed | `set TORCH_CUDA_ARCH_LIST=12.0` (or your arch) before the run |
| CUDA OOM at `factor 2` | full-resolution images | keep `--factor 4`, or lower `--max_gaussians` |
| H-PSNR is NaN for some views | marking projected outside the frame | expected at extreme altitudes; those views are excluded (`n = 33`) |
