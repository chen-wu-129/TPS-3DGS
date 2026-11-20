# --- open-source path shim (added for release) ---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import REPO_ROOT, DATA_ROOT, OUT_ROOT, RESULTS_ROOT  # noqa: E402
# -------------------------------------------------
# -*- coding: utf-8 -*-
"""eval_dose_v4.py —— ROI 权重剂量曲线（倒 U 闭环）

held-out（288 训练 / 41 验证，26 个 H 可见视角），固定 salience=0（纯 ROI 加权 L1，
与 decompose 的 lossonly 同协议），在 w=1(base)/2/3(lossonly)/4/5 五档上评测：
  全图 PSNR/SSIM、H 区(4m 圆) PSNR/SSIM、总高斯数；并画 H-PSNR 对 w 的倒 U 曲线。
输出：
  D:\低空视景\04_指标数据\v4_dose_curve_holdout.csv
  D:\低空视景\04_指标数据\v4_dose_curve_holdout.png
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).parent))
from mini_train import load_scene, make_window, psnr, ssim
from eval_altitude import load_ply_splats, render_view
from read_altitude import read_xmp

OUTD = RESULTS_ROOT
# 权重 -> 模型目录。w=1 必须用 train_ours 同代码路径的不加权参照（budget，W=1），
# 不能用 mini_train 的 base，否则跨脚本 0.3~0.5dB 系统差会压低起点、夸大剂量净增益。
DOSE = [(1, "budget"), (2, "w2"), (3, "lossonly"), (4, "w4"), (5, "w5l")]


def count_gaussians(ply):
    return PlyData.read(ply)["vertex"].count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(DATA_ROOT, "fieldH"))
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--max_images", type=int, default=331)
    ap.add_argument("--sh_degree", type=int, default=2)
    ap.add_argument("--root", default=os.path.join(OUT_ROOT, "v4_fieldH"))
    ap.add_argument("--h_radius_m", type=float, default=4.0)
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

    def eval_ply(ply):
        s = load_ply_splats(ply, args.sh_degree, device)
        rows = []
        with torch.no_grad():
            for idx in val_ids:
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
                r4 = Kn[0, 0] * (args.h_radius_m / scale) / pc[2]
                if not (0 <= u <= w and 0 <= v <= h) or r4 < 2:
                    continue
                yy, xx = np.mgrid[0:h, 0:w]
                hm = ((xx - u) ** 2 + (yy - v) ** 2) < r4 ** 2
                yidx, xidx = np.where(hm)
                y0, y1 = max(yidx.min(), 0), min(yidx.max() + 1, h)
                x0, x1 = max(xidx.min(), 0), min(xidx.max() + 1, w)
                sub = hm[y0:y1, x0:x1]
                hp = pred[y0:y1, x0:x1][sub]
                hg = gt[y0:y1, x0:x1][sub]
                mse_h = ((hp - hg) ** 2).mean().item()
                hpsnr = -10.0 * np.log10(max(mse_h, 1e-12))
                hssim = ssim(
                    pred[y0:y1, x0:x1].permute(2, 0, 1)[None],
                    gt[y0:y1, x0:x1].permute(2, 0, 1)[None], window).mean().item()
                rows.append(dict(psnr=psnr(pred, gt).item(),
                                 ssim=ssim(pred.permute(2, 0, 1)[None],
                                           gt.permute(2, 0, 1)[None],
                                           window).mean().item(),
                                 hpsnr=hpsnr, hssim=hssim))
        del s
        import gc; gc.collect(); torch.cuda.empty_cache()
        return rows

    table = []
    for w, cfg in DOSE:
        ply = os.path.join(args.root, f"seed0_{cfg}", "mini_trained.ply")
        if not os.path.exists(ply):
            print("MISSING", ply); continue
        rv = eval_ply(ply)
        ng = count_gaussians(ply)
        table.append(dict(w=w, cfg=cfg, n=len(rv),
                          psnr=np.mean([r["psnr"] for r in rv]),
                          ssim=np.mean([r["ssim"] for r in rv]),
                          hpsnr=np.mean([r["hpsnr"] for r in rv]),
                          hssim=np.mean([r["hssim"] for r in rv]),
                          ngauss=ng))
        print(f"w={w} ({cfg}) n={len(rv)} H-PSNR={table[-1]['hpsnr']:.3f} "
              f"H-SSIM={table[-1]['hssim']:.4f} N={ng:,}")

    if len(table) < 2:
        print("可用模型不足，无法画剂量曲线"); return

    csv_path = os.path.join(OUTD, "v4_dose_curve_holdout.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["w", "config", "n_views", "PSNR_full", "SSIM_full",
                     "H_PSR", "H_SSIM", "n_gaussians"])
        for t in table:
            wr.writerow([t["w"], t["cfg"], t["n"],
                         f"{t['psnr']:.3f}", f"{t['ssim']:.4f}",
                         f"{t['hpsnr']:.3f}", f"{t['hssim']:.4f}", t["ngauss"]])

    try:
        import os as _os
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        for _fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
            if _os.path.exists(_fp):
                font_manager.fontManager.addfont(_fp)
                matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=_fp).get_name()
                break
        matplotlib.rcParams["axes.unicode_minus"] = False
        ws = [t["w"] for t in table]
        hp = [t["hpsnr"] for t in table]
        fp = [t["psnr"] for t in table]
        plt.figure(figsize=(7.2, 4.6), dpi=190)
        plt.plot(ws, hp, "o-", color="#2f5f9e", lw=2.4, ms=8, label="标志区 H-PSNR")
        plt.plot(ws, fp, "s--", color="#888888", lw=1.6, ms=6, label="全图 PSNR")
        for x, y in zip(ws, hp):
            plt.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                         xytext=(0, 9), ha="center", fontsize=9)
        plt.xlabel("ROI 权重 w（纯损失加权 L，固定预算）")
        plt.ylabel("PSNR / dB")
        plt.xticks(ws)
        plt.title("ROI 权重剂量响应（held-out，33 个标志入画视角；w=1 为同代码路径不加权参照）")
        plt.grid(alpha=0.3)
        plt.legend(loc="lower right", fontsize=10)
        plt.tight_layout()
        png_path = os.path.join(REPO_ROOT, "budget_fig/v4_dose_curve_holdout.png")
        plt.savefig(png_path)
        print("SAVED", png_path)
    except Exception as e:
        print("绘图失败（不影响 CSV）:", e)
    print("SAVED", csv_path)
    print("DOSE_EVAL_DONE")


if __name__ == "__main__":
    main()
