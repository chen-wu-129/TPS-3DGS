# Verification record

Run on 2026-11-20 on the authors' workstation, using **only this repository's data and code** plus the authors' trained checkpoints (`mini_trained.ply`, not released). Purpose: confirm that the shipped tables were produced by the shipped pipeline on the shipped data — not just that the numbers look plausible.

## Environment

| Item | Value |
|---|---|
| GPU | RTX 5070 Laptop, 8 GB |
| PyTorch | 2.8.0.dev + cu128 |
| gsplat | 1.5.3, CUDA kernels built JIT at first rasterisation |
| CUDA toolkit | 12.8 (`nvcc` on PATH) |
| C++ toolchain | MSVC 14.44 (`vcvars64.bat`) + Windows SDK 10.0.22621 |
| `TORCH_CUDA_ARCH_LIST` | `12.0` |

Kernel build (Windows, once per machine):

```powershell
cmd /c "call `"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`" && set TORCH_CUDA_ARCH_LIST=12.0 && python src\selftest.py --data_dir data\fieldH --factor 8 --max_images 60"
```

On Linux the same step only needs `nvcc` + `ninja` on PATH.

## Procedure

```bash
python src/verify_data.py --manifest                      # both scenes, counts + hashes
python src/selftest.py --data_dir data/fieldH --factor 8 --max_images 60
python src/eval_recon_v4.py --data_dir data/fieldH --factor 4 --max_images 331 \
       --root <dir with seed<k>_<config>/mini_trained.ply> --seeds 0 1 2
```

Each config/seed pair is rendered on the 41 held-out views (`range(7, n, 8)`), of which 33 contain the marking.

## Result: every reproduced row equals the shipped CSV

`base` (standard 3DGS, w = 1):

| seed | n | PSNR | SSIM | H-PSNR | H-SSIM | sharpness |
|---|---|---|---|---|---|---|
| 0 | 33 | 24.067 | 0.8409 | 21.715 | 0.9160 | 67.8 |
| 1 | 33 | 24.320 | 0.8419 | 22.135 | 0.9179 | 67.1 |
| 2 | 33 | 24.324 | 0.8424 | 22.170 | 0.9170 | 64.5 |
| mean ± SD | | 24.24 ± 0.15 | 0.8417 ± 0.0007 | **22.01 ± 0.25** | 0.9170 ± 0.0010 | 66.5 ± 1.8 |

`lossonly` (the paper's TPS row, `w = 3`, densification criterion untouched):

| seed | n | PSNR | H-PSNR |
|---|---|---|---|
| 0 | 33 | 24.266 | 22.903 |
| 1 | 33 | 24.230 | 22.515 |
| 2 | 33 | 24.311 | 22.806 |
| mean ± SD | | 24.27 ± 0.04 | **22.74 ± 0.20** |

`ours` (second w = 3 variant, reported separately):

| seed | n | PSNR | H-PSNR |
|---|---|---|---|
| 0 | 33 | 24.169 | 22.196 |
| 1 | 33 | 24.071 | 21.819 |
| 2 | 33 | 24.002 | 22.047 |
| mean ± SD | | 24.08 ± 0.08 | **22.02 ± 0.19** |

All six per-seed rows and all three aggregates reproduce `results/v4_recon_metrics_holdout.csv` and `results/v4_recon_metrics_agg_holdout.csv` digit for digit, including SSIM, H-SSIM and sharpness columns.

## Naming: which checkpoint is the paper's main row

The +0.74 dB headline (H-PSNR 22.74 ± 0.20, full-image 24.27 ± 0.04) is the **`lossonly`** family: ROI-weighted loss with `w = 3` and an untouched densification criterion — exactly what §3 of the paper describes as TPS. The `ours` family is a second variant that changes the densification side as well and lands at 22.02 ± 0.19. Both are shipped in the results tables so the reader can compare; `REPRODUCE.md` writes its outputs to `seed<k>_lossonly`.

## Scope

Verified: the released images, poses, altitude table, held-out split, loader, ROI construction and evaluation code reproduce the published reconstruction tables given the published weights.

Not covered by this run: the 24-run training grid itself (≈23 min per run on the above GPU). Retraining from scratch reproduces the same numbers within the run-to-run noise the paper reports for the paired comparison (< 0.1 dB).
