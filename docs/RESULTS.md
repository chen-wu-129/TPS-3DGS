# What is in `results/`

One file per experiment, all raw — no hand-edited values. Column names are unchanged from the scripts that wrote them, so `sample` and `清晰度%` are Chinese for "number of views" and "sharpness retention (%)".

## Main comparison

| File | Paper element | Content |
|---|---|---|
| `v4_recon_metrics_holdout.csv` | Table 5 (raw) | one row per config × seed: full-image PSNR/SSIM, marking-region H-PSNR/H-SSIM, sharpness ratio, n = 33 views |
| `v4_recon_metrics_agg_holdout.csv` | Table 5 | the same aggregated as mean ± SD over three seeds |
| `v4_stats_tests_holdout.csv` | Table 5 note, §5.1 | means, paired t-tests, bootstrap CIs, Cohen's d |
| `v4_ply_stats_holdout.csv` | Table 5, in-ROI Gaussians | total / in-ROI counts, ROI density, size and opacity statistics per seed |
| `v4_budget_agg.csv` | §5.1 budget analysis | net Gaussian change inside and outside the ROI over the densification window |

## Where the gain comes from

| File | Paper element | Content |
|---|---|---|
| `v4_grad_probe_holdout.csv` | Table 1 | screen-space criterion gradient in the marking region, per altitude: `gH_w` = ROI-weighted, `gO_w` = background, `ratio_h` vs. the theoretical prediction |
| `v5_grad_accum_w3.csv` | Table 2 | accumulated gradient state per decision step at `w = 3`: `n_over_tau`, `n_over_split_elig`, `n_over_clone_elig`, quantiles normalised by τ |
| `v4_event_counts_phys_holdout.csv` | Table 3 | clone/split counts inside 1.5 / 2 / 3 / 4 / 6 m physical radii, baseline vs. TPS — the split-suppression evidence |
| `v4_radius_scan_holdout.csv` | Table 8, Fig. 11 | H-PSNR as the evaluation radius tightens from 6 m to 1.5 m, all configs |
| `v4_seam_band_holdout.csv` | Table 8 | full-image and concentric-band PSNR (0–4, 4–6, 6–10, 10–14 m) — the +0.83 dB seam-band result |
| `v4_dose_curve_holdout.csv` | Fig. 10 | `w = 1…5` sweep: full PSNR, H-PSNR, Gaussian count — the inverted-U peaking at `w = 3` |
| `v4_stress_hpsnr_holdout.csv` | robustness check | H-PSNR under Gaussian blur (σ 5 / 11) and sensor noise (0.01 / 0.03), baseline vs. TPS |

## Supporting material

| File | Content |
|---|---|
| `metrics_extra.txt` | retrain statistics on the 329-view model: SSIM/H-SSIM per altitude tier, sharpness retention, training time, peak memory, Gaussian counts (baseline / w3 / w5) |
| `fps_out.txt` | rendering FPS log (same scene, 329 views loaded) |
| `jxfx_multi_metrics.csv` | secondary scene (`jxfx`), lower altitude: full-image PSNR/SSIM and marking PSNR at 1.5 / 2.5 m radii, per view |

## Regenerating a table

Scripts are in `src/` (training/evaluation) and `src/analysis/` (statistics, probes, plots). Every command below assumes the checkpoints from [`REPRODUCE.md`](REPRODUCE.md) §2 exist under `outputs/v4_fieldH/`; pass `--root` to point elsewhere.

| Result file | Paper element | Script | Command |
|---|---|---|---|
| `v4_recon_metrics_holdout.csv` | Table 5 (raw) | `src/eval_recon_v4.py` | `python src/eval_recon_v4.py --data_dir data/fieldH --factor 4 --root outputs/v4_fieldH --seeds 0 1 2` |
| `v4_recon_metrics_agg_holdout.csv` | Table 5 (mean ± SD) | `src/aggregate_v4.py` | `python src/aggregate_v4.py --root outputs/v4_fieldH` |
| `v4_stats_tests_holdout.csv` | Table 5 note, §5.1 | `src/analysis/stats_test_v4.py` | `python src/analysis/stats_test_v4.py` (config list at the top of the file) |
| `v4_ply_stats_holdout.csv` | Table 5, in-ROI Gaussians | `src/analysis/budget_stats_v4.py` | `python src/analysis/budget_stats_v4.py --root outputs/v4_fieldH --seeds 0 1 2` |
| `v4_budget_agg.csv` | §5.1 budget analysis | `src/analysis/budget_stats_v4.py` | same command (it writes both files) |
| `v4_grad_probe_holdout.csv` | Table 1 | `src/analysis/grad_probe_v4.py` | `python src/analysis/grad_probe_v4.py --model base` / `--model ours --weight 3` |
| (Table 1 figure) | Fig. 9 | `src/analysis/replot_grad_probe.py` | `python src/analysis/replot_grad_probe.py` |
| `v5_grad_accum_w3.csv` | Table 2 | `src/train_v5.py` | `python src/train_v5.py --data_dir data/fieldH --h_roi_weight 3 …` (same protocol as `train_ours.py`, with the per-step counters logged) |
| `v4_event_counts_phys_holdout.csv` | Table 3 | `src/analysis/sum_evt_phys.py` | `python src/analysis/sum_evt_phys.py` (reads the `event_trace.csv` files written by `--event_log`) |
| `v4_seam_band_holdout.csv` | Table 8, bands | `src/eval_altitude.py` | `python src/eval_altitude.py --data_dir data/fieldH --ply outputs/v4_fieldH/seed0_ours/mini_trained.ply --factor 4` |
| `v4_radius_scan_holdout.csv` | Table 8, Fig. 11 | `src/analysis/eval_radius_scan_v4.py` | `python src/analysis/eval_radius_scan_v4.py --configs base ours salonly` |
| `v4_dose_curve_holdout.csv` | Fig. 10 | `src/analysis/eval_dose_v4.py` | `python src/analysis/eval_dose_v4.py --h_radius_m 4` |
| `v4_stress_hpsnr_holdout.csv` | robustness | `src/analysis/eval_stress_v4.py` | `python src/analysis/eval_stress_v4.py --configs base ours` |
| `metrics_extra.txt` | §5.1 retrain statistics | `src/eval_recon_v4.py` | written alongside the main evaluation |
| `fps_out.txt` | §5.3 rendering budget | `src/analysis/render_fps.py` | `python src/analysis/render_fps.py` |
| `jxfx_multi_metrics.csv` | Table 10 | `src/analysis/eval_jxfx_multi.py` | `python src/analysis/eval_jxfx_multi.py` (second scene in `data/jxfx`, models `seed<0..2>_base` / `seed<0..2>_lossonly` under `outputs/jxfx/`) |

Files with no command-line interface (`stats_test_v4.py`, `sum_evt_phys.py`, `render_fps.py`, `eval_jxfx_multi.py`) carry their paths and config lists as constants at the top of the file; edit them in place or set `TPS_DATA_ROOT` / `TPS_OUT_ROOT` / `TPS_RESULTS_ROOT`.

## Reading the numbers honestly

- **Seeds.** `base`, `ours` and `salonly` are three-seed runs. `focal`, `randroi`, `randfix`, `gradscale` and the radius/band scans are single-seed (`seed 0`) exploratory runs — the paper says so in the table notes, and the ± column is absent for those rows.
- **Δ sign.** `ΔH-PSNR` is always relative to the three-seed baseline mean of 22.01 dB, except the Focal Loss row, which is paired on seed 0 (20.296 − 21.715 = −1.42 dB).
- **Rounding.** Deltas are computed from unrounded values, so a column may not add up exactly from the rounded entries (the +0.83 dB seam band is one such case).
- **n = 33.** H-PSNR is undefined when the marking projects outside the frame; the evaluation filters on that and reports `n` per row.
- **`budget`** in `v4_dose_curve_holdout.csv` is the raised Gaussian cap (2× the default), used to show that capacity is not the bottleneck; it is not the same as the Local Densification-Only baseline.
- **Second scene (Table 10).** `jxfx_multi_metrics.csv` holds the per-view values behind Table 10 (14 held-out views × 3 seeds × 2 configs). The scene itself is released as `data/jxfx` (117 images, poses, altitude), so the table is reproducible once `seed<k>_base` and `seed<k>_lossonly` are trained there; the 33 GB raw capture is not redistributed (identifiable campus site), only the frames actually used.
