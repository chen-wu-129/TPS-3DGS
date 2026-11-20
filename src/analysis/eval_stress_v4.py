# --- open-source path shim (added for release) ---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import REPO_ROOT, DATA_ROOT, OUT_ROOT, RESULTS_ROOT  # noqa: E402
# -------------------------------------------------
# -*- coding: utf-8 -*-
"""eval_stress_v4.py —— 任务61（合成退化压力测试）+ 任务63（ROI 接缝环带）

加载训练好的 PLY，渲染 held-out 的 26 个 H 视角（val_ids=range(7,n,8)），评测：
  1) 合成退化压力测试：对 pred/GT 施加相同运动模糊(轻/重)或独立高斯噪声(轻/重)，
     比较各配置 H-PSNR 在退化下相对 base 的增益符号是否保持（鲁棒性）。
  2) ROI 接缝环带：H 区(0-4m)、过渡带(4-6m)、接缝带(6-10m)、外围(10-14m) 的 PSNR，
     排除"标志变清晰但加权区边界出现接缝"。
默认在 results/v4_holdout（seed0）上跑；--all_train 切到 v4_fieldH（3 seeds）。
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from mini_train import load_scene, make_window, psnr, ssim
from eval_altitude import load_ply_splats, render_view
from read_altitude import read_xmp

OUTD = RESULTS_ROOT
RINGS_M = [4.0, 6.0, 10.0, 14.0]  # H区 / 过渡带 / 接缝带 / 外围


def motion_kernel(L, angle_deg=0.0):
    k = np.zeros((L, L), dtype=np.float32)
    c = L // 2
    a = np.deg2rad(angle_deg)
    for t in range(-c, c + 1):
        x = int(round(c + t * np.cos(a)))
        y = int(round(c + t * np.sin(a)))
        if 0 <= x < L and 0 <= y < L:
            k[y, x] = 1.0
    s = k.sum()
    return k / s if s > 0 else k


def apply_blur(img01, L, angle=0.0):
    k = motion_kernel(L, angle)
    x = (img01.detach().cpu().numpy() * 255.0).astype(np.float32)
    y = cv2.filter2D(x, -1, k)
    return torch.from_numpy(y / 255.0).clamp(0, 1)


def apply_noise(img01, sigma, seed):
    rng = np.random.default_rng(seed)
    x = (img01.detach().cpu().numpy() * 255.0).astype(np.float32)
    y = x + rng.normal(0, sigma * 255.0, x.shape).astype(np.float32)
    return torch.from_numpy(np.clip(y, 0, 255) / 255.0)


def psnr_pixels(a, b):
    mse = ((a - b) ** 2).mean().item()
    return -10.0 * np.log10(max(mse, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(DATA_ROOT, "fieldH"))
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--max_images", type=int, default=331)
    ap.add_argument("--sh_degree", type=int, default=2)
    ap.add_argument("--root", default=os.path.join(OUT_ROOT, "v4_fieldH"))
    ap.add_argument("--tag", default="_holdout")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--configs", nargs="*",
                    default=["base", "lossonly", "salonly", "ours", "w5",
                             "focal", "randroi", "randfix", "gradscale", "budget"])
    args = ap.parse_args()
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
    mm = ~np.isnan(alt)
    b_axis = 2
    A = np.polyfit(cam[mm, b_axis], alt[mm], 1)
    scale = abs(A[0])
    from h3_util import correct_h3
    H3 = correct_h3(cam, pts, hax=(0, 1), b_axis=b_axis)
    window = make_window(device)

    def ring_masks(w, h, u, v, K, pc2):
        yy, xx = np.mgrid[0:h, 0:w]
        d = np.sqrt((xx - u) ** 2 + (yy - v) ** 2)
        rpx = [K[0, 0] * (rm / scale) / pc2 for rm in RINGS_M]
        masks = {
            "H_0-4m": d < rpx[0],
            "trans_4-6m": (d >= rpx[0]) & (d < rpx[1]),
            "seam_6-10m": (d >= rpx[1]) & (d < rpx[2]),
            "outer_10-14m": (d >= rpx[2]) & (d < rpx[3]),
        }
        return masks

    def eval_ply(ply):
        s = load_ply_splats(ply, args.sh_degree, device)
        per_view = []
        with torch.no_grad():
            for vi, idx in enumerate(val_ids):
                gt = data["pixels"][idx].to(device)
                w, h = data["wh"][idx]
                c2w = c2w_all[idx:idx + 1].to(device)
                K = data["Ks"][idx:idx + 1].to(device)
                pred, _ = render_view(s, c2w, K, w, h, args.sh_degree)
                viewmat = torch.linalg.inv(c2w)[0].cpu().numpy()
                pc = viewmat @ np.array([H3[0], H3[1], H3[2], 1.0])
                if pc[2] <= 0:
                    continue
                Kn = K[0].cpu().numpy()
                u = Kn[0, 0] * pc[0] / pc[2] + Kn[0, 2]
                v = Kn[1, 1] * pc[1] / pc[2] + Kn[1, 2]
                r4 = Kn[0, 0] * (4.0 / scale) / pc[2]
                if not (0 <= u <= w and 0 <= v <= h) or r4 < 2:
                    continue
                rec = dict(alt=float(alt[idx]))

                # 退化条件下的 H 区（0-4m 圆）PSNR
                masks = ring_masks(w, h, u, v, Kn, pc[2])
                hm = masks["H_0-4m"]
                yidx, xidx = np.where(hm)
                y0, y1 = max(yidx.min(), 0), min(yidx.max() + 1, h)
                x0, x1 = max(xidx.min(), 0), min(xidx.max() + 1, w)
                sub_hm = hm[y0:y1, x0:x1]

                variants = {
                    "clean": (pred, gt),
                    "blur_l5": (apply_blur(pred, 5), apply_blur(gt, 5)),
                    "blur_h11": (apply_blur(pred, 11), apply_blur(gt, 11)),
                    "noise_s01": (apply_noise(pred, 0.01, idx * 7 + 1),
                                 apply_noise(gt, 0.01, idx * 7 + 2)),
                    "noise_s03": (apply_noise(pred, 0.03, idx * 7 + 3),
                                 apply_noise(gt, 0.03, idx * 7 + 4)),
                }
                for cond, (p, g) in variants.items():
                    pn = p.cpu().numpy() if p.device.type != "cpu" else p.numpy()
                    gn = g.cpu().numpy() if g.device.type != "cpu" else g.numpy()
                    p_s = pn[y0:y1, x0:x1][sub_hm]
                    g_s = gn[y0:y1, x0:x1][sub_hm]
                    rec[f"hpsnr_{cond}"] = psnr_pixels(
                        torch.tensor(p_s), torch.tensor(g_s))

                # clean 全图 / H 区 SSIM（矩形 crop）
                rec["psnr_full"] = psnr(pred, gt).item()
                rec["ssim_full"] = ssim(pred.permute(2, 0, 1)[None],
                                        gt.permute(2, 0, 1)[None], window).mean().item()
                rec["ssim_H"] = ssim(
                    pred[y0:y1, x0:x1].permute(2, 0, 1)[None],
                    gt[y0:y1, x0:x1].permute(2, 0, 1)[None], window).mean().item()

                # 接缝环带（clean，mask 像素 PSNR）
                pn = pred.cpu().numpy()
                gn = gt.cpu().numpy()
                for rname, msk in masks.items():
                    rec[f"psnr_{rname}"] = psnr_pixels(
                        torch.tensor(pn[msk]), torch.tensor(gn[msk]))
                per_view.append(rec)
        del s
        import gc; gc.collect(); torch.cuda.empty_cache()
        return per_view

    results = {}
    for cfg in args.configs:
        ply = os.path.join(args.root, f"seed{args.seed}_{cfg}", "mini_trained.ply")
        if not os.path.exists(ply):
            print("MISSING", ply); continue
        results[cfg] = eval_ply(ply)
        print(f"{cfg:<10} n={len(results[cfg])} 渲染完成")

    if not results:
        print("无可用模型"); return

    # ---- 任务61：退化压力测试聚合 ----
    conds = ["clean", "blur_l5", "blur_h11", "noise_s01", "noise_s03"]
    spath = os.path.join(OUTD, f"v4_stress_hpsnr{args.tag}.csv")
    base_h = {c: np.mean([r[f"hpsnr_{c}"] for r in results["base"]])
              if "base" in results else None for c in conds}
    with open(spath, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["config"] + [f"H-PSNR_{c}" for c in conds]
                     + [f"delta_{c}" for c in conds])
        for cfg, rv in results.items():
            hv = [np.mean([r[f"hpsnr_{c}"] for r in rv]) for c in conds]
            dv = [hv[i] - base_h[c] if base_h[c] is not None else np.nan
                  for i, c in enumerate(conds)]
            wr.writerow([cfg] + [f"{x:.3f}" for x in hv] + [f"{x:+.3f}" for x in dv])
    print("\n===== 合成退化压力测试（H-PSNR，括号为相对 base 的 Δ）=====")
    header = f"{'config':<10}" + "".join(f"{c:>14}" for c in conds)
    print(header)
    for cfg, rv in results.items():
        line = f"{cfg:<10}"
        for c in conds:
            v = np.mean([r[f"hpsnr_{c}"] for r in rv])
            d = v - base_h[c] if base_h[c] is not None else np.nan
            line += f"{v:>7.2f}{d:>+5.2f}"
        print(line)

    # ---- 任务63：接缝环带 ----
    ring_names = ["H_0-4m", "trans_4-6m", "seam_6-10m", "outer_10-14m"]
    rpath = os.path.join(OUTD, f"v4_seam_band{args.tag}.csv")
    with open(rpath, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["config", "psnr_full", "ssim_full", "ssim_H"]
                     + [f"psnr_{x}" for x in ring_names])
        for cfg, rv in results.items():
            wr.writerow([cfg]
                        + [f"{np.mean([r['psnr_full'] for r in rv]):.3f}",
                           f"{np.mean([r['ssim_full'] for r in rv]):.4f}",
                           f"{np.mean([r['ssim_H'] for r in rv]):.4f}"]
                        + [f"{np.mean([r[f'psnr_{x}'] for r in rv]):.3f}"
                           for x in ring_names])
    print("\n===== ROI 接缝环带 PSNR（clean）=====")
    print(f"{'config':<10}{'full':>8}{'H区':>8}{'过渡4-6':>9}{'接缝6-10':>10}{'外围10-14':>11}")
    for cfg, rv in results.items():
        print(f"{cfg:<10}{np.mean([r['psnr_full'] for r in rv]):>8.2f}"
              f"{np.mean([r['psnr_H_0-4m'] for r in rv]):>8.2f}"
              f"{np.mean([r['psnr_trans_4-6m'] for r in rv]):>9.2f}"
              f"{np.mean([r['psnr_seam_6-10m'] for r in rv]):>10.2f}"
              f"{np.mean([r['psnr_outer_10-14m'] for r in rv]):>11.2f}")
    print("\nSAVED", spath)
    print("SAVED", rpath)
    print("STRESS_EVAL_DONE")


if __name__ == "__main__":
    main()
