# --- open-source path shim (added for release) ---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import REPO_ROOT, DATA_ROOT, OUT_ROOT, RESULTS_ROOT  # noqa: E402
# -------------------------------------------------
# -*- coding: utf-8 -*-
"""eval_jxfx_multi.py —— 江飞字牌第二场景：三种子配对统计（机制边界负对照）。

口径与 stats_test_v4.py 完全一致：
  - 逐 (seed, view) 配对差（lossonly - base）
  - 配对 t 检验、按 seed 分层 bootstrap 95% CI、Cohen's d、每 seed 方向
  - 指标：全图 PSNR / SSIM、字牌区 r=1.5m 与 r=2.5m M-PSNR
held-out 视角 val_ids=range(7,117,8)=14；只统计字牌投影落在画面内的视角（相机几何与 seed 无关）。
"""
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, REPO_ROOT)
from mini_train import load_scene, make_window, psnr, ssim  # noqa: E402
from eval_altitude import load_ply_splats, render_view  # noqa: E402
from read_altitude import read_xmp  # noqa: E402
from scipy import stats  # noqa: E402

DATA = os.path.join(DATA_ROOT, "jxfx")
ROOT = os.path.join(OUT_ROOT, "jxfx")
H3 = np.array([0.001681, 0.028397, -0.001327])
RADII = [1.5, 2.5]
SEEDS = [0, 1, 2]
N_BOOT = 10000
OUT_CSV = os.path.join(RESULTS_ROOT, "jxfx_multi_metrics.csv")
OUT_STATS = os.path.join(RESULTS_ROOT, "jxfx_multi_stats.csv")


def main():
    device = "cuda"
    data = load_scene(DATA, 4, 117)
    n = len(data["pixels"])
    val_ids = list(range(7, n, 8))
    cam = data["camtoworlds"][:, :3, 3].numpy()
    alt = []
    for nm in data["names"]:
        rel, _, _ = read_xmp(os.path.join(DATA, "images", nm))
        alt.append(rel if rel is not None else np.nan)
    alt = np.array(alt)
    m = ~np.isnan(alt)
    A = np.polyfit(cam[m, 2], alt[m], 1)
    scale = abs(A[0])
    print(f"scale={scale:.3f} m/unit, val 视角数={len(val_ids)}")
    window = make_window(device)

    # 字牌入画视角（相机几何相同，与 seed 无关）
    keep_idx = []
    for idx in val_ids:
        c2w = data["camtoworlds"][idx:idx + 1].to(device)
        viewmat = torch.linalg.inv(c2w)[0].cpu().numpy()
        pc = viewmat @ np.array([H3[0], H3[1], H3[2], 1.0])
        K = data["Ks"][idx].cpu().numpy()
        w, h = data["wh"][idx]
        if pc[2] > 0:
            q = K @ pc[:3]
            u, v = q[0] / q[2], q[1] / q[2]
            if 0 <= u <= w and 0 <= v <= h:
                keep_idx.append(idx)
    print("字牌入画视角数:", len(keep_idx))

    def eval_ply(ply):
        s = load_ply_splats(ply, 2, device)
        full, ssfull = [], []
        mp = {R: [] for R in RADII}
        with torch.no_grad():
            for idx in keep_idx:
                gt = data["pixels"][idx].to(device)
                w, h = data["wh"][idx]
                c2w = data["camtoworlds"][idx:idx + 1].to(device)
                K = data["Ks"][idx:idx + 1].to(device)
                pred, _ = render_view(s, c2w, K, w, h, 2)
                full.append(psnr(pred, gt).item())
                ssfull.append(ssim(pred.permute(2, 0, 1)[None],
                                   gt.permute(2, 0, 1)[None], window).mean().item())
                Kn = K[0].cpu().numpy()
                viewmat = torch.linalg.inv(c2w)[0].cpu().numpy()
                pc = viewmat @ np.array([H3[0], H3[1], H3[2], 1.0])
                q = Kn @ pc[:3]
                u, v = q[0] / q[2], q[1] / q[2]
                for R in RADII:
                    rpx = Kn[0, 0] * (R / scale) / pc[2]
                    x0, x1 = int(max(0, u - rpx)), int(min(w, u + rpx))
                    y0, y1 = int(max(0, v - rpx)), int(min(h, v + rpx))
                    val = np.nan
                    if x1 - x0 > 3 and y1 - y0 > 3:
                        val = psnr(pred[y0:y1, x0:x1], gt[y0:y1, x0:x1]).item()
                    mp[R].append(val)
        del s
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        return np.array(full), np.array(ssfull), {R: np.array(mp[R]) for R in RADII}

    res = {"base": {}, "lossonly": {}}
    for model in ["base", "lossonly"]:
        for sd in SEEDS:
            ply = os.path.join(ROOT, f"seed{sd}_{model}", "mini_trained.ply")
            if not os.path.exists(ply):
                print("MISSING", ply)
                continue
            full, ss, mp = eval_ply(ply)
            res[model][sd] = (full, ss, mp)
            print(f"{model:<9} s{sd} full={full.mean():.3f} ss={ss.mean():.4f} "
                  f"mp1.5={mp[1.5].mean():.3f} mp2.5={mp[2.5].mean():.3f}")

    seeds_ok = [sd for sd in SEEDS if sd in res["base"] and sd in res["lossonly"]]
    print("可配对 seeds:", seeds_ok)
    rng = np.random.default_rng(42)

    def paired(getter):
        d, per_seed = [], []
        for sd in seeds_ok:
            dd = getter(res["lossonly"][sd]) - getter(res["base"][sd])
            d.append(dd)
            per_seed.append(dd.mean())
        d = np.concatenate(d)
        per_seed = np.array(per_seed)
        t, p = stats.ttest_rel(d, np.zeros_like(d))
        dcohen = d.mean() / (d.std(ddof=1) + 1e-12)
        boot = []
        for _ in range(N_BOOT):
            vals = []
            for sd in seeds_ok:
                dd = getter(res["lossonly"][sd]) - getter(res["base"][sd])
                bi = rng.integers(0, len(dd), len(dd))
                vals.append(dd[bi].mean())
            boot.append(np.mean(vals))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return d.mean(), p, lo, hi, dcohen, per_seed

    metrics = [
        ("全图 PSNR", lambda x: x[0]),
        ("全图 SSIM", lambda x: x[1]),
        ("字牌 r=1.5m M-PSNR", lambda x: x[2][1.5]),
        ("字牌 r=2.5m M-PSNR", lambda x: x[2][2.5]),
    ]

    # 每模型跨 seed 均值±SD（先取每 seed 内视角均值，再跨 seed）
    print("\n===== 各模型三种子均值±SD =====")
    summary_rows = []
    for model in ["base", "lossonly"]:
        for name, getter in metrics:
            xs = np.array([getter(res[model][sd]).mean() for sd in seeds_ok])
            print(f"{model:<9} {name:<22} {xs.mean():.4f} ± {xs.std(ddof=1):.4f}")
            summary_rows.append([model, name, f"{xs.mean():.4f}", f"{xs.std(ddof=1):.4f}"])

    print("\n===== 配对检验（lossonly - base）=====")
    stat_rows = []
    for name, getter in metrics:
        dm, p, lo, hi, dcohen, per_seed = paired(getter)
        if np.all(per_seed > 0):
            direction = "全正"
        elif np.all(per_seed < 0):
            direction = "全负"
        else:
            direction = "混合"
        print(f"{name:<22} delta={dm:+.4f}  p={p:.4f}  CI[{lo:+.4f},{hi:+.4f}] "
              f"d={dcohen:.2f} 每seed {np.round(per_seed, 4)} ({direction})")
        stat_rows.append([name, f"{dm:+.4f}", f"{p:.4f}", f"[{lo:+.4f},{hi:+.4f}]",
                          f"{dcohen:.2f}", direction,
                          ";".join(f"{x:.4f}" for x in per_seed)])

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["model", "seed", "view_idx", "alt", "psnr_full",
                     "ssim_full", "mpsnr_1.5", "mpsnr_2.5"])
        for model in ["base", "lossonly"]:
            for sd in seeds_ok:
                full, ss, mp = res[model][sd]
                for j, idx in enumerate(keep_idx):
                    wr.writerow([model, sd, idx, round(alt[idx], 1),
                                 round(full[j], 4), round(ss[j], 4),
                                 round(mp[1.5][j], 4), round(mp[2.5][j], 4)])

    with open(OUT_STATS, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["# 三种子均值±SD"])
        wr.writerow(["model", "metric", "mean", "sd"])
        wr.writerows(summary_rows)
        wr.writerow([])
        wr.writerow(["# 配对检验（lossonly - base）"])
        wr.writerow(["metric", "delta", "p", "95CI", "cohen_d", "seed_direction", "per_seed_delta"])
        wr.writerows(stat_rows)

    print("\nSAVED", OUT_CSV)
    print("SAVED", OUT_STATS)
    print("JXFX_MULTI_EVAL_DONE")


if __name__ == "__main__":
    main()
