# --- open-source path shim (added for release) ---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import REPO_ROOT, DATA_ROOT, OUT_ROOT, RESULTS_ROOT  # noqa: E402
# -------------------------------------------------
# -*- coding: utf-8 -*-
"""eval_recon_v4.py —— 7 配置 x seeds 的重建侧质量评测（26 真实视角）

加载场景一次，循环渲染各 PLY 的 held-out 视角（val_ids = range(7,n,8)），
口径同 eval_real_altitude.py：
  全图 PSNR/SSIM + H 区 H-PSNR/H-SSIM + 清晰度保持率（H 投影半径 4 m）
输出 v4_recon_metrics.csv（每模型一行）+ 分档明细。
注意：v4 模型由 --all_train 训练，本评测为训练视角内拟合质量，论文需如实标注。
"""
import argparse, csv, os, sys
from pathlib import Path
import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from mini_train import load_scene, make_window, psnr, ssim
from eval_altitude import load_ply_splats, render_view
from read_altitude import read_xmp

ROOT = os.path.join(OUT_ROOT, "v4_fieldH")
OUTD = RESULTS_ROOT
CONFIGS = ["base", "ours", "w5", "focal", "randroi", "gradscale", "budget"]
DECOMPOSE_CONFIGS = ["base", "lossonly", "salonly", "ours", "w5",
                     "focal", "randroi", "randfix", "gradscale", "budget"]
SEEDS = [0, 1, 2]
EDGES = [27, 40, 55, 75, 100, 120]


def laplacian_var(img01):
    g = (img01.detach().cpu().numpy() * 255.0).mean(axis=2).astype(np.float32)
    return float(cv2.Laplacian(np.ascontiguousarray(g), cv2.CV_32F).var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(DATA_ROOT, "fieldH"))
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--max_images", type=int, default=331)
    ap.add_argument("--sh_degree", type=int, default=2)
    ap.add_argument("--h_radius_m", type=float, default=4.0)
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--tag", default="")
    ap.add_argument("--extra", action="store_true", help="加入 w2/w4/r3/r9 消融配置")
    ap.add_argument("--decompose", action="store_true", help="2x2 机制拆解矩阵")
    args = ap.parse_args()
    root = args.root
    tag = args.tag
    seeds = args.seeds
    if args.decompose:
        configs = DECOMPOSE_CONFIGS
    else:
        configs = CONFIGS + (["w2", "w4", "r3", "r9"] if args.extra else [])
    device = "cuda"

    print("加载场景...")
    data = load_scene(args.data_dir, args.factor, args.max_images)
    n = len(data["pixels"])
    val_ids = list(range(7, n, 8))
    c2w_all = data["camtoworlds"]
    cam = c2w_all[:, :3, 3].numpy()
    pts = data["points"].numpy()

    alt = []
    for nm in data["names"]:
        rel, _, _ = read_xmp(os.path.join(args.data_dir, "images", nm))
        alt.append(rel if rel is not None else np.nan)
    alt = np.array(alt)
    m = ~np.isnan(alt)
    b_axis = 2
    A = np.polyfit(cam[m, b_axis], alt[m], 1)
    scale_m_per_unit = abs(A[0])
    from h3_util import correct_h3
    H3 = correct_h3(cam, pts, hax=(0, 1), b_axis=b_axis)
    print(f"高度拟合 1单位≈{scale_m_per_unit:.1f}m, val视角数={len(val_ids)}")

    window = make_window(device)
    W, H = data["wh"][0] if isinstance(data["wh"][0], int) else (int(data["wh"][0][0]), int(data["wh"][0][1]))

    def eval_ply(ply):
        s = load_ply_splats(ply, args.sh_degree, device)
        rows = []
        with torch.no_grad():
            for idx in val_ids:
                gt = data["pixels"][idx].to(device)
                w, h = data["wh"][idx]
                c2w = c2w_all[idx:idx + 1].to(device)
                K = data["Ks"][idx:idx + 1].to(device)
                pred, alpha = render_view(s, c2w, K, w, h, args.sh_degree)
                p = psnr(pred, gt).item()
                ss = ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], window).mean().item()
                viewmat = torch.linalg.inv(c2w)[0].cpu().numpy()
                pc = viewmat @ np.array([H3[0], H3[1], H3[2], 1.0])
                Kn = K[0].cpu().numpy()
                hpsnr = hssim = sharp = np.nan
                if pc[2] > 0:
                    q = Kn @ pc[:3]
                    u, v = q[0] / q[2], q[1] / q[2]
                    r_norm = args.h_radius_m / scale_m_per_unit
                    rpx = Kn[0, 0] * r_norm / pc[2]
                    x0, x1 = int(max(0, u - rpx)), int(min(w, u + rpx))
                    y0, y1 = int(max(0, v - rpx)), int(min(h, v + rpx))
                    if x1 - x0 > 3 and y1 - y0 > 3 and 0 <= u <= w and 0 <= v <= h:
                        hpsnr = psnr(pred[y0:y1, x0:x1], gt[y0:y1, x0:x1]).item()
                        hssim = ssim(pred[y0:y1, x0:x1].permute(2, 0, 1)[None],
                                     gt[y0:y1, x0:x1].permute(2, 0, 1)[None], window).mean().item()
                        sp = laplacian_var(pred[y0:y1, x0:x1])
                        sg = laplacian_var(gt[y0:y1, x0:x1])
                        sharp = sp / max(sg, 1e-6) * 100.0
                rows.append(dict(alt=float(alt[idx]), psnr=p, ssim=ss,
                                 hpsnr=hpsnr, hssim=hssim, sharp=sharp))
        del s
        import gc; gc.collect(); torch.cuda.empty_cache()
        return rows

    def agg(rows):
        keep = [r for r in rows if not np.isnan(r["hpsnr"])]
        out = dict(n=len(keep),
                   psnr=np.mean([r["psnr"] for r in keep]),
                   ssim=np.mean([r["ssim"] for r in keep]),
                   hpsnr=np.mean([r["hpsnr"] for r in keep]),
                   hssim=np.mean([r["hssim"] for r in keep]),
                   sharp=np.mean([r["sharp"] for r in keep]))
        bands = []
        for i in range(len(EDGES) - 1):
            sel = [r for r in keep if EDGES[i] <= r["alt"] < EDGES[i + 1]]
            if sel:
                bands.append((len(sel), np.mean([r["hpsnr"] for r in sel]),
                              np.mean([r["hssim"] for r in sel]),
                              np.mean([r["sharp"] for r in sel])))
            else:
                bands.append((0, np.nan, np.nan, np.nan))
        return out, bands

    detail, summary = [], {}
    for cfg in configs:
        summary[cfg] = []
        for sd in args.seeds:
            ply = os.path.join(root, f"seed{sd}_{cfg}", "mini_trained.ply")
            if not os.path.exists(ply):
                print("MISSING", ply); continue
            rows = eval_ply(ply)
            o, bands = agg(rows)
            summary[cfg].append((o, bands))
            detail.append((cfg, sd, o))
            print(f"{cfg:<9} s{sd} n={o['n']:2d} PSNR={o['psnr']:.2f} SSIM={o['ssim']:.4f} "
                  f"H-PSNR={o['hpsnr']:.2f} H-SSIM={o['hssim']:.4f} 清晰度={o['sharp']:.1f}%")

    # CSV
    cpath = os.path.join(OUTD, f"v4_recon_metrics{tag}.csv")
    with open(cpath, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["config", "seed", "n", "PSNR", "SSIM", "H-PSNR", "H-SSIM", "清晰度%"])
        for cfg, sd, o in detail:
            w.writerow([cfg, sd, o["n"], f"{o['psnr']:.3f}", f"{o['ssim']:.4f}",
                        f"{o['hpsnr']:.3f}", f"{o['hssim']:.4f}", f"{o['sharp']:.1f}"])
    apath = os.path.join(OUTD, f"v4_recon_metrics_agg{tag}.csv")
    with open(apath, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["config", "PSNR", "SSIM", "H-PSNR", "H-SSIM", "清晰度%"])
        for cfg in configs:
            if not summary[cfg]:
                continue
            def ms(k):
                xs = [o[k] for o, _ in summary[cfg]]
                return np.mean(xs), (np.std(xs, ddof=1) if len(xs) > 1 else 0.0)
            pm, psd = ms("psnr"); sm, ssd = ms("ssim"); hm, hsd = ms("hpsnr")
            hsm, hssd = ms("hssim"); shm, shsd = ms("sharp")
            w.writerow([cfg, f"{pm:.2f}±{psd:.2f}", f"{sm:.4f}±{ssd:.4f}",
                        f"{hm:.2f}±{hsd:.2f}", f"{hsm:.4f}±{hssd:.4f}", f"{shm:.1f}±{shsd:.1f}"])

    print("\n===== 3 seeds 均值±SD =====")
    print(f"{'config':<10}{'PSNR':>14}{'SSIM':>16}{'H-PSNR':>14}{'H-SSIM':>16}{'清晰度%':>12}")
    for cfg in configs:
        if not summary[cfg]:
            continue
        def ms(k):
            xs = [o[k] for o, _ in summary[cfg]]
            return np.mean(xs), (np.std(xs, ddof=1) if len(xs) > 1 else 0.0)
        pm, psd = ms("psnr"); sm, ssd = ms("ssim"); hm, hsd = ms("hpsnr")
        hsm, hssd = ms("hssim"); shm, shsd = ms("sharp")
        print(f"{cfg:<10}{pm:>8.2f}±{psd:<4.2f}{sm:>8.4f}±{ssd:<6.4f}{hm:>8.2f}±{hsd:<4.2f}"
              f"{hsm:>8.4f}±{hssd:<6.4f}{shm:>6.1f}±{shsd:<4.1f}")
    print("\nSAVED", cpath)
    print("SAVED", apath)
    print("EVAL_RECON_V4_DONE")


if __name__ == "__main__":
    main()
