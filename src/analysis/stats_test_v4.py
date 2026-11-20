# --- open-source path shim (added for release) ---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import REPO_ROOT, DATA_ROOT, OUT_ROOT, RESULTS_ROOT  # noqa: E402
# -------------------------------------------------
# -*- coding: utf-8 -*-
"""stats_test_v4.py —— 核心四配置 × 3 seeds 的配对统计检验

逐视角（26 个 H 在画面内 held-out 视角）计算全图 PSNR、H-PSNR，按 (seed,view) 配对：
  - 各配置 3 seeds 均值±SD
  - lossonly/salonly/ours vs base：配对 t 检验、按 seed 分层 bootstrap 95% CI、Cohen's d
  - 总高斯数（PLY 点数）3 seeds 均值±SD 与百分比
输出 v4_stats_tests_holdout.csv 与 per-view 明细，并打印结论。
"""
import csv, os, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from mini_train import load_scene, make_window, psnr, ssim
from eval_altitude import load_ply_splats, render_view
from read_altitude import read_xmp
from plyfile import PlyData
from scipy import stats

ROOT = os.path.join(OUT_ROOT, "v4_fieldH")
OUTD = RESULTS_ROOT
# base 为标准 3DGS（mini_train）；budget 为同一扩展训练脚本下的 W=1 不加权参照（同代码路径）；
# 其余三个为干预配置。配对检验同时报告 vs base（相对标准 3DGS）与 vs budget（同代码路径净效应）。
CONFIGS = ["base", "budget", "lossonly", "salonly", "ours"]
METHOD = ["lossonly", "salonly", "ours"]
SEEDS = [0, 1, 2]
H_RAD = 4.0
N_BOOT = 10000


def ply_count(ply):
    return len(PlyData.read(ply)["vertex"])


def main():
    data = load_scene(os.path.join(DATA_ROOT, "fieldH"), 4, 331)
    n = len(data["pixels"])
    val_ids = list(range(7, n, 8))
    c2w_all = data["camtoworlds"]
    cam = c2w_all[:, :3, 3].numpy()
    pts = data["points"].numpy()
    alt = []
    for nm in data["names"]:
        rel, _, _ = read_xmp(os.path.join(os.path.join(DATA_ROOT, "fieldH"), "images", nm))
        alt.append(rel if rel is not None else np.nan)
    alt = np.array(alt)
    m = ~np.isnan(alt)
    A = np.polyfit(cam[m, 2], alt[m], 1)
    scale = abs(A[0])
    from h3_util import correct_h3
    H3 = correct_h3(cam, pts, hax=(0, 1), b_axis=2)
    device = "cuda"
    window = make_window(device)

    # 仅保留 H 在画面内视角，所有配置共用同一集合
    keep_idx = []
    for idx in val_ids:
        c2w = c2w_all[idx:idx + 1].to(device)
        viewmat = torch.linalg.inv(c2w)[0].cpu().numpy()
        pc = viewmat @ np.array([H3[0], H3[1], H3[2], 1.0])
        if pc[2] > 0:
            K = data["Ks"][idx].cpu().numpy()
            q = K @ pc[:3]; u, v = q[0] / q[2], q[1] / q[2]
            w, h = data["wh"][idx]
            if 0 <= u <= w and 0 <= v <= h:
                keep_idx.append(idx)
    print("H 在画面内视角数:", len(keep_idx))

    # 采集 {cfg: {seed: (full_psnr[], hpsnr[], n_gauss)}}
    res = {c: {} for c in CONFIGS}
    rng = np.random.default_rng(42)
    for cfg in CONFIGS:
        for sd in SEEDS:
            ply = os.path.join(ROOT, f"seed{sd}_{cfg}", "mini_trained.ply")
            if not os.path.exists(ply):
                print("MISSING", ply); continue
            ng = ply_count(ply)
            s = load_ply_splats(ply, 2, device)
            full, hp = [], []
            with torch.no_grad():
                for idx in keep_idx:
                    gt = data["pixels"][idx].to(device)
                    w, h = data["wh"][idx]
                    c2w = c2w_all[idx:idx + 1].to(device)
                    K = data["Ks"][idx:idx + 1].to(device)
                    pred, _ = render_view(s, c2w, K, w, h, 2)
                    full.append(psnr(pred, gt).item())
                    Kn = K[0].cpu().numpy()
                    viewmat = torch.linalg.inv(c2w)[0].cpu().numpy()
                    pc = viewmat @ np.array([H3[0], H3[1], H3[2], 1.0])
                    q = Kn @ pc[:3]; u, v = q[0] / q[2], q[1] / q[2]
                    rpx = Kn[0, 0] * (H_RAD / scale) / pc[2]
                    x0, x1 = int(max(0, u - rpx)), int(min(w, u + rpx))
                    y0, y1 = int(max(0, v - rpx)), int(min(h, v + rpx))
                    if x1 - x0 > 3 and y1 - y0 > 3:
                        hp.append(psnr(pred[y0:y1, x0:x1], gt[y0:y1, x0:x1]).item())
            res[cfg][sd] = (np.array(full), np.array(hp), ng)
            print(f"{cfg:<9} s{sd} n_gauss={ng:,} full={np.mean(full):.2f} H={np.mean(hp):.2f}")
            del s
            import gc; gc.collect(); torch.cuda.empty_cache()

    seeds_ok = [sd for sd in SEEDS if all(sd in res[c] for c in CONFIGS)]
    print("可用于配对的 seeds:", seeds_ok)

    def ms_metric(cfg, key):
        # key: 0 full,1 H
        arr = [np.mean(res[cfg][sd][key]) for sd in seeds_ok]
        return np.mean(arr), np.std(arr, ddof=1) if len(arr) > 1 else 0.0, arr

    def paired(cfg, key, ref):
        # 逐 (seed,view) 配对差（ref 指定参照配置）
        d = []
        per_seed = []
        for sd in seeds_ok:
            dd = res[cfg][sd][key] - res[ref][sd][key]
            d.append(dd); per_seed.append(dd.mean())
        d = np.concatenate(d)
        per_seed = np.array(per_seed)
        t, p = stats.ttest_rel(d, np.zeros_like(d))
        dcohen = d.mean() / (d.std(ddof=1) + 1e-12)
        # 分层 bootstrap（seed 内重采样视角）
        boot = []
        for _ in range(N_BOOT):
            vals = []
            for sd in seeds_ok:
                dd = res[cfg][sd][key] - res[ref][sd][key]
                idx = rng.integers(0, len(dd), len(dd))
                vals.append(dd[idx].mean())
            boot.append(np.mean(vals))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return d.mean(), p, lo, hi, dcohen, per_seed

    rows = []
    print("\n===== 重建质量 3 seeds 均值±SD =====")
    for cfg in CONFIGS:
        fm, fsd, _ = ms_metric(cfg, 0); hm, hsd, _ = ms_metric(cfg, 1)
        ng = np.mean([res[cfg][sd][2] for sd in seeds_ok])
        print(f"{cfg:<9} full {fm:.2f}±{fsd:.2f} | H {hm:.2f}±{hsd:.2f} | n_gauss {ng:,.0f}")
        rows.append([cfg, f"{fm:.3f}±{fsd:.3f}", f"{hm:.3f}±{hsd:.3f}",
                     f"{ng:,.0f}"])

    stat_rows = []
    for ref, tag in [("base", "vs标准基线base"), ("budget", "vs同代码路径W1(budget)")]:
        print(f"\n===== {tag} 配对检验（H-PSNR / 全图 PSNR）=====")
        for cfg in METHOD:
            for key, name in [(1, "H-PSNR"), (0, "full-PSNR")]:
                dm, p, lo, hi, dcohen, per_seed = paired(cfg, key, ref)
                allpos = "是" if np.all(per_seed > 0) else "否"
                print(f"{cfg:<9} {name:<9} Δ={dm:+.3f} dB  p={p:.2e}  95%CI[{lo:+.3f},{hi:+.3f}]  "
                      f"d={dcohen:.2f}  每seed方向一致={allpos} {np.round(per_seed,3)}")
                stat_rows.append([tag, cfg, name, f"{dm:+.3f}", f"{p:.4f}",
                                  f"[{lo:+.3f},{hi:+.3f}]", f"{dcohen:.2f}", allpos])

    # 总高斯数比较（两个参照口径）
    print("\n===== 总高斯数（3 seeds 均值）与配对检验 =====")
    for ref, tag in [("base", "vs标准基线base"), ("budget", "vs同代码路径W1(budget)")]:
        nb = np.mean([res[ref][sd][2] for sd in seeds_ok])
        for cfg in METHOD:
            nc = np.mean([res[cfg][sd][2] for sd in seeds_ok])
            pct = 100.0 * (nc - nb) / nb
            if len(seeds_ok) > 1:
                _, pn = stats.ttest_rel([res[cfg][sd][2] for sd in seeds_ok],
                                        [res[ref][sd][2] for sd in seeds_ok])
            else:
                pn = float("nan")
            print(f"[{tag}] {cfg:<9} n={nc:,.0f}  相对参照 {pct:+.2f}%  p={pn:.3f}")
            stat_rows.append([tag, cfg, "n_gauss", f"{nc:,.0f}", f"{pn:.3f}",
                              f"{pct:+.2f}%", "", ""])

    with open(os.path.join(OUTD, "v4_stats_tests_holdout.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["# 重建质量 3 seeds 均值±SD"])
        w.writerow(["config", "full-PSNR", "H-PSNR", "n_gauss"])
        w.writerows(rows)
        w.writerow([])
        w.writerow(["# 配对检验（vs base 标准口径 / vs budget 同代码路径口径）"])
        w.writerow(["reference", "config", "metric", "delta", "p", "95%CI", "cohen_d", "all_seeds_positive"])
        w.writerows(stat_rows)
    print("\nSAVED", os.path.join(OUTD, "v4_stats_tests_holdout.csv"))
    print("STATS_TEST_DONE")


if __name__ == "__main__":
    main()
