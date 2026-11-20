# TPS-3DGS

**Targeted Photometric Supervision for small-target 3D Gaussian Splatting — VTOL landing-pad markings in low-altitude visual scenes.**

3D Gaussian Splatting reconstructs landing-pad markings poorly: strokes bleed across their boundaries, take the colour of the surrounding turf, and fragment. We trace this to **Photometric Supervision Dilution (PSD)** — the image-averaged loss divides every residual by the total pixel count `HW`, so the normalized screen-space gradient reaching the marking region stays at `0.04τ–0.32τ`, a factor of 3–25 below the densification threshold, and the primitives inside the marking are neither fitted nor refined. **TPS** projects an altitude-adaptive screen-space ROI analytically from the positional prior of vertiport markings and reweights the loss inside it, leaving the renderer, the densification criterion and the inference cost untouched. With `w = 3` the marking-region H-PSNR improves by **+0.74 dB** (95 % CI [+0.58, +0.90], p < 0.0001, consistent across three seeds) with no significant change in full-image PSNR (`+0.03 dB`, p = 0.63).

This repository contains the code, the flight images and the sparse reconstruction of the single-scene study, plus every result table behind the paper.

## What is released

| Part | Path | Content |
|---|---|---|
| Code | `src/` | training (baseline / TPS / controls), evaluation, figure scripts |
| Flight data | `data/fieldH/` | 331 DJI Air 3S images (2048×1152), COLMAP text model, altitude table |
| Result tables | `results/` | all per-configuration metrics, statistics and ablations |
| Docs | `docs/` | reproduction protocol, results walk-through |

Not included: trained `.ply` models (1–2 GB each, regenerable with the commands below).

## Repository layout

```
TPS-3DGS/
├── src/
│   ├── mini_train.py           # standard 3DGS baseline (w = 1)
│   ├── train_ours.py           # TPS and all controls (ROI weighting / salience / focal / randroi / randfix / gradscale)
│   ├── read_altitude.py        # altitude source (XMP, with altitude.csv fallback)
│   ├── h3_util.py              # vertical-axis and metric-scale estimation
│   ├── normalize.py            # scene normalisation used by the loader
│   ├── selftest.py             # GPU-free sanity check (data, altitude, H localisation, ROI projection)
│   ├── verify_data.py          # data integrity + MANIFEST hashes for both scenes
│   ├── eval_altitude.py        # full-image + marking-region metrics
│   ├── eval_recon_v4.py        # held-out evaluation driver (PSNR / SSIM / H-PSNR / H-SSIM)
│   ├── eval_synth_closed.py    # closed-loop synthetic-view evaluation
│   ├── render_synthetic_v2.py  # synthetic viewpoint rendering
│   ├── aggregate_v4.py         # per-view → per-config aggregation, paired tests
│   ├── paths.py                # single source of truth for every path
│   ├── imgio.py                # image reader that survives non-ASCII paths (Windows)
│   ├── train_v5.py             # training with per-step gradient counters (Table 2)
│   ├── analysis/               # probes, statistics and plots behind the tables
│   └── figures/                # figure scripts (problem statement, dose–response, budget, collection)
├── data/fieldH/
│   ├── images/                 # 331 JPG, 2048×1152, EXIF/XMP stripped
│   ├── sparse/0/               # cameras.txt, images.txt, points3D.txt (COLMAP text)
│   └── altitude.csv            # relative altitude per image (metres above take-off)
├── data/jxfx/                  # second scene (Table 10): 117 JPG + sparse/0 + altitude.csv
├── results/                    # CSV tables quoted in the paper
├── docs/REPRODUCE.md           # step-by-step reproduction
└── docs/RESULTS.md             # what each table means
```

## Environment

Verified on Python 3.9 + PyTorch 2.8 (CUDA 12.8) + `gsplat` 1.5.3, single RTX 5070 Laptop (8 GB).

```bash
pip install -r requirements.txt
```

`gsplat` needs a CUDA toolchain at run time (its kernels are JIT-compiled on first use). On Linux install `nvcc` + `ninja`; on Windows install the MSVC build tools and run inside a `vcvars64` shell. Set `TORCH_CUDA_ARCH_LIST` to your GPU architecture if the automatic detection fails.

## Quick start

All paths below are relative to the repository root, and every script writes into `outputs/` (override with `TPS_OUT_ROOT`, `TPS_DATA_ROOT`, `TPS_RESULTS_ROOT`).

```bash
# 0) data integrity: image/pose/point/altitude counts and the held-out split, per scene
python src/verify_data.py            # both scenes, no GPU needed

# 0b) loader + altitude + H localisation + ROI projection
python src/selftest.py --data_dir data/fieldH --factor 8 --max_images 60

# 1) standard 3DGS baseline, 30k iterations, 12k densification window
python src/mini_train.py --data_dir data/fieldH --factor 4 --max_images 331 \
    --max_steps 30000 --sh_degree 2 --refine_stop 12000 \
    --out outputs/v4_fieldH/seed0_base

# 2) TPS (w = 3): loss reweighted inside the projected marking ROI, criterion untouched
python src/train_ours.py --data_dir data/fieldH --factor 4 --max_images 331 \
    --max_steps 30000 --sh_degree 2 --refine_stop 12000 \
    --loss_mode roi --h_roi_weight 3.0 --h_radius_m 6.0 --salience_strength 0 \
    --out outputs/v4_fieldH/seed0_ours

# 3) dose–response sweep (w = 5)
python src/train_ours.py --data_dir data/fieldH --loss_mode roi --h_roi_weight 5.0 \
    --salience_strength 0 --out outputs/v4_fieldH/seed0_w5

# 4) controls: focal loss, random-drift ROI, gradient scaling
python src/train_ours.py --data_dir data/fieldH --loss_mode focal     --out outputs/v4_fieldH/seed0_focal
python src/train_ours.py --data_dir data/fieldH --loss_mode randroi   --out outputs/v4_fieldH/seed0_randroi
python src/train_ours.py --data_dir data/fieldH --loss_mode gradscale --h_roi_weight 3.0 --out outputs/v4_fieldH/seed0_gradscale

# 5) evaluate all configurations (held-out views = every 8th image) and aggregate
python src/eval_recon_v4.py --data_dir data/fieldH --root outputs/v4_fieldH --seeds 0 1 2
python src/aggregate_v4.py
```

Folder names matter: `eval_recon_v4.py` looks for `<root>/seed<k>_<config>/mini_trained.ply`, with `config` in `base, ours, w5, focal, randroi, randfix, gradscale, budget` (`--decompose` adds `lossonly, salonly`).

Control configurations used in the paper:

| Folder suffix | Flags | Paper name |
|---|---|---|
| `base` | `mini_train.py` | Standard baseline (w = 1) |
| `ours` | `--loss_mode roi --h_roi_weight 3 --salience_strength 0` | **TPS (w = 3)** |
| `w5` | `--h_roi_weight 5 --salience_strength 0` | weight sweep |
| `salonly` | `--h_roi_weight 1 --salience_strength 3 --salience_radius_m 8` | Local Densification-Only (−0.25 dB) |
| `focal` | `--loss_mode focal` | Focal Loss (−1.42 dB) |
| `randroi` | `--loss_mode randroi` | Random-drift ROI (+0.23 dB) |
| `randfix` | `--loss_mode randfix --randfix_offset_m 25` | Fixed-offset ROI (+0.10 dB) |
| `gradscale` | `--loss_mode gradscale --h_roi_weight 3` | Gradient scaling (−0.14 dB) |

Random seeds: `--seed 0/1/2`. `--factor 4` halves each image side (2048×1152 → 1024×576); use `--factor 2` for full resolution if you have the memory.

## Data

`data/fieldH` is one orbital flight (**20 Oct 2026, a vertiport in Jiangxi, China**) captured with a DJI Air 3S: 331 images of a white "H" marking on turf, 329 registered by COLMAP (the two landing frames failed), relative altitude 21.9–93.7 m (absolute 23.4–118.0 m). The released images are re-encoded 2048×1152 copies with all EXIF/XMP/GPS metadata stripped, so the altitude used by the altitude-adaptive ROI comes from `data/fieldH/altitude.csv`; `read_altitude.py` transparently falls back to that table.

Evaluation split used throughout the paper: every 8th image (41 views) held out, the remaining 288 images for training; 33 of the 41 held-out views contain the marking.

- **Marking evaluation region (H-PSNR):** the projected square of 4 m object-space radius centred on the marking; concentric bands are reported in 0–4 / 4–6 / 6–10 / 10–14 m.
- **Metric scale:** recovered from the altitude table by regressing camera height against relative altitude; `h3_util.py` and `train_ours.py` estimate the vertical axis the same way, so the ROI follows altitude automatically.
- **Two scenes.** `data/fieldH` (the `H` marking, all main tables) and `data/jxfx` (the campus sign, Table 10): 117 images with poses and an altitude table each. Not released: the trained checkpoints — reproduce them with §2 of [`docs/REPRODUCE.md`](docs/REPRODUCE.md) (≈23 min per run).

## Results

Three seeds, 33 held-out views in which the marking appears (mean ± SD):

| Configuration | Full-image PSNR | Marking-region H-PSNR | In-ROI Gaussians | ΔH-PSNR |
|---|---|---|---|---|
| Standard baseline (w = 1) | 24.24 ± 0.15 | 22.01 ± 0.25 | 6,700 ± 130 | — |
| **TPS (w = 3)** | 24.27 ± 0.04 | **22.74 ± 0.20** | 10,279 ± 556 | **+0.74** [+0.58, +0.90], p < 0.0001 |
| Local Densification-Only | 24.15 ± 0.15 | 21.76 ± 0.08 | 15,975 ± 321 | −0.25 [−0.43, −0.07], p = 0.013 |

Supporting evidence (see `docs/RESULTS.md` for the full walk-through and `results/` for raw numbers):

- criterion gradient in the marking region stays within `0.04τ–0.32τ` at every altitude and training stage (3–25× below threshold);
- split-eligible instances rise from 167 to 369 at `w = 3`, while split events are only 0.42–0.75 % of all events in the 0–2 m bands;
- with densification frozen after iteration 15,000, weighting alone still returns **+0.272 dB**;
- gain is monotone inward: +1.16 dB in the 0–4 m core, +1.23 dB in the 4–6 m transition band, +0.83 dB in the 6–10 m seam band, +0.55 dB at 10–14 m;
- a spatially misaligned ROI keeps only +0.23 dB (random drift) or +0.10 dB (fixed 25 m offset), and pure gradient scaling −0.14 dB — localization, not reweighting alone, carries the gain.

Training cost per 30k-iteration run at `factor 4`: ≈23 min, ≈2.0 GB peak, ≈1.99 M Gaussians on an 8 GB laptop GPU.

## Citation

```bibtex
@article{wu2026tps3dgs,
  title   = {Targeted Photometric Supervision for Small-Target Reconstruction of
             VTOL Landing-Pad Markings with 3D Gaussian Splatting},
  author  = {Wu, Chen and others},
  journal = {The Visual Computer},
  year    = {2026},
  note    = {Code and data: https://github.com/chen-wu-129/TPS-3DGS}
}
```

See `CITATION.cff` for the machine-readable entry (update it once the paper's DOI is final).

## License

Code: MIT (`LICENSE`). Flight images and the sparse reconstruction in `data/`: CC BY 4.0 — please cite the paper if you use them.

## Contact

Questions about the code or the data: open an issue, or write to the corresponding author listed on the paper.
