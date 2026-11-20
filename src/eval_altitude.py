# --- open-source path shim (added for release) ---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import REPO_ROOT, DATA_ROOT, OUT_ROOT, RESULTS_ROOT  # noqa: E402
# -------------------------------------------------
# -*- coding: utf-8 -*-
"""
eval_altitude.py —— 高度分层评测（论文第一个实验：高度 vs 渲染质量）
训练和评测分离：本脚本加载训练好的 PLY，对每个测试视角渲染并算 PSNR/SSIM，
再按"归一化高度 z"和"相机距离"分层统计，输出 CSV + 曲线图。

用法：
  python eval_altitude.py --data_dir "./data/fieldH/images" \
      --ply results\fuyao_full\mini_trained.ply \
      --factor 8 --max_images 1095 --sh_degree 2 --out results\fuyao_full
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent / "ref"))
from mini_train import (  # noqa: E402
    load_scene, make_window, psnr, ssim,
)


def load_ply_splats(ply_path, sh_degree, device):
    """从 PLY 恢复 gsplat 高斯参数（scales 为 log、opacities 为 logit）"""
    from plyfile import PlyData
    v = PlyData.read(ply_path)["vertex"]
    n = len(v)
    means = np.stack([v["x"], v["y"], v["z"]], 1)
    scales = np.stack([v[f"scale_{i}"] for i in range(3)], 1)
    quats = np.stack([v[f"rot_{i}"] for i in range(4)], 1)
    opac = v["opacity"]
    sh0 = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1)[:, None, :]
    rest_names = [p.name for p in v.properties if p.name.startswith("f_rest_")]
    rest_names = sorted(rest_names, key=lambda s: int(s.split("_")[2]))
    K = (sh_degree + 1) ** 2 - 1
    shN = np.stack([v[name] for name in rest_names[:K * 3]], 1).reshape(n, K, 3)
    t = lambda a: torch.from_numpy(np.asarray(a)).float().to(device)
    return {
        "means": t(means), "scales": t(scales), "quats": t(quats),
        "opacities": t(opac), "sh0": t(sh0), "shN": t(shN),
    }


def render_view(s, c2w, K, w, h, sh_degree):
    from gsplat import rasterization
    colors = torch.cat([s["sh0"], s["shN"]], 1)
    renders, alphas, info = rasterization(
        means=s["means"], quats=s["quats"],
        scales=torch.exp(s["scales"]),
        opacities=torch.sigmoid(s["opacities"]),
        colors=colors, viewmats=torch.linalg.inv(c2w), Ks=K,
        width=w, height=h, sh_degree=sh_degree,
    )
    return renders[0].clamp(0, 1), alphas[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--factor", type=int, default=8)
    ap.add_argument("--max_images", type=int, default=1095)
    ap.add_argument("--sh_degree", type=int, default=2)
    ap.add_argument("--n_bins", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda"
    os.makedirs(args.out, exist_ok=True)

    print("加载场景...")
    data = load_scene(args.data_dir, args.factor, args.max_images)
    n = len(data["pixels"])
    val_ids = list(range(7, n, 8))  # 与训练完全一致的测试划分
    print(f"测试视角 {len(val_ids)} 个；加载 PLY...")
    s = load_ply_splats(args.ply, args.sh_degree, device)
    window = make_window(device)

    # 归一化后的相机位置（z 为高度）
    c2w_all = data["camtoworlds"]
    cam_pos = c2w_all[:, :3, 3]
    height = cam_pos[:, 2]
    center = cam_pos.mean(0)
    dist = torch.linalg.norm(cam_pos - center, dim=1)

    rows = []
    with torch.no_grad():
        for vi, idx in enumerate(val_ids):
            gt = data["pixels"][idx].to(device)
            w, h = data["wh"][idx]
            c2w = c2w_all[idx:idx + 1].to(device)
            K = data["Ks"][idx:idx + 1].to(device)
            pred, alpha = render_view(s, c2w, K, w, h, args.sh_degree)
            p = psnr(pred, gt).item()
            ss = ssim(pred.permute(2, 0, 1)[None],
                      gt.permute(2, 0, 1)[None], window).mean().item()
            cover = alpha.mean().item()
            rows.append({
                "view": data["names"][idx],
                "height": float(height[idx]),
                "dist": float(dist[idx]),
                "psnr": p, "ssim": ss, "alpha_cover": cover,
            })
            if (vi + 1) % 20 == 0:
                print(f"  评测 {vi+1}/{len(val_ids)}")

    csv_path = os.path.join(args.out, "per_view_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"每视角指标: {csv_path}")

    heights = np.array([r["height"] for r in rows])
    psnrs = np.array([r["psnr"] for r in rows])
    ssims = np.array([r["ssim"] for r in rows])

    # 按高度分 n_bins 层（按分位数切，保证每层都有样本）
    edges = np.percentile(heights, np.linspace(0, 100, args.n_bins + 1))
    edges[-1] += 1e-6
    print("\n===== 高度分层（按分位数）=====")
    layer_stat = []
    for i in range(args.n_bins):
        m = (heights >= edges[i]) & (heights < edges[i + 1])
        if m.sum() == 0:
            continue
        layer_stat.append({
            "layer": i, "h_mid": 0.5 * (edges[i] + edges[i + 1]),
            "n": int(m.sum()),
            "psnr": psnrs[m].mean(), "ssim": ssims[m].mean(),
        })
        print(f"  L{i+1}  高度 {edges[i]:6.2f}~{edges[i+1]:6.2f}  "
              f"n={m.sum():3d}  PSNR {psnrs[m].mean():6.2f}  SSIM {ssims[m].mean():.4f}")
    print(f"整体 PSNR {psnrs.mean():.2f} / SSIM {ssims.mean():.4f}")

    # 画高度-质量曲线
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
        if os.path.exists(fp):
            font_manager.fontManager.addfont(fp)
            matplotlib.rcParams["font.family"] = \
                font_manager.FontProperties(fname=fp).get_name()
            break
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, ax1 = plt.subplots(figsize=(8, 5))
    hm = [l["h_mid"] for l in layer_stat]
    pmean = [l["psnr"] for l in layer_stat]
    smean = [l["ssim"] for l in layer_stat]
    ax1.plot(hm, pmean, "o-", color="#c0392b", lw=2, ms=8, label="PSNR")
    ax1.set_xlabel("归一化飞行高度（z，越大越高）")
    ax1.set_ylabel("PSNR / dB", color="#c0392b")
    ax1.tick_params(axis="y", labelcolor="#c0392b")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(hm, smean, "s--", color="#2471a3", lw=2, ms=8, label="SSIM")
    ax2.set_ylabel("SSIM", color="#2471a3")
    ax2.tick_params(axis="y", labelcolor="#2471a3")
    plt.title("飞行高度 vs 渲染质量（fuyao 测试视角）")
    fig.tight_layout()
    fig_path = os.path.join(args.out, "altitude_quality.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"高度-质量曲线图: {fig_path}")

    # 同时画距离-质量（Octree-GS 对照）
    fig2, ax = plt.subplots(figsize=(8, 5))
    dists = np.array([r["dist"] for r in rows])
    ax.scatter(dists, psnrs, c=heights, cmap="turbo", s=22, alpha=0.85)
    cb = plt.colorbar(ax.collections[0], ax=ax)
    cb.set_label("高度 z")
    ax.set_xlabel("相机到场景中心距离")
    ax.set_ylabel("PSNR / dB")
    ax.set_title("距离/高度 vs PSNR（每个点一个测试视角）")
    ax.grid(alpha=0.3)
    fig2.tight_layout()
    fig2_path = os.path.join(args.out, "dist_quality.png")
    plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
    print(f"距离-质量散点图: {fig2_path}")


if __name__ == "__main__":
    main()
