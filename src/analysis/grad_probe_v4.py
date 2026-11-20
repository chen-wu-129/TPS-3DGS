# --- open-source path shim (added for release) ---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import REPO_ROOT, DATA_ROOT, OUT_ROOT, RESULTS_ROOT  # noqa: E402
# -------------------------------------------------
# -*- coding: utf-8 -*-
"""grad_probe_v4.py —— 任务62（屏幕梯度随高度衰减+临界高度）+ 任务64（加权/不加权同视角实测）

在致密化窗口内的 ckpt（默认 step 10000）上，对每个 held-out 视角分别做
"标准 L1" 与 "ROI 加权 L1" 的 forward-backward，直接取 gsplat 致密化判据使用的
屏幕梯度（means2d 的 absgrad，按 width/2、height/2 归一化），按 H 区/背景分组：

  任务62：base 标准 L1 下 H 区平均屏幕梯度随高度衰减，叠加阈值 grow_grad2d=0.0002，
          给出标准致密化在 H 区失效的临界高度；并与几何预测 r_px∝1/d 对照。
  任务64：同一模型同一视角，加权/标准 的 H 区梯度比值（单视角）随高度，
          对照纯归一化理论 w/[1+(w-1)p]；再给跨视角（多视角）聚合比值。

探针只测 L1 项（0.8 系数在比值中抵消），不改变任何模型。
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from mini_train import load_scene
from eval_altitude import load_ply_splats
from train_ours import roi_weight_map
from read_altitude import read_xmp

OUTD = RESULTS_ROOT
GROW_GRAD2D = 0.0002


def estimate_h3d(data, data_dir):
    cam = data["camtoworlds"][:, :3, 3].numpy()
    pts = data["points"].numpy()
    alt = []
    for nm in data["names"]:
        rel, _, _ = read_xmp(os.path.join(data_dir, "images", nm))
        alt.append(rel if rel is not None else np.nan)
    alt = np.array(alt)
    m = ~np.isnan(alt)
    b_axis = int(np.argmax([abs(np.corrcoef(cam[m, a], alt[m])[0, 1]) for a in range(3)]))
    A = np.polyfit(cam[m, b_axis], alt[m], 1)
    scale = abs(A[0])
    from h3_util import correct_h3
    H3 = correct_h3(cam, pts, hax=(0, 1), b_axis=b_axis)
    return H3, scale, alt


def screen_grad(splats, c2w, K, w, h, gt, sh_degree, Wmap=None):
    """单次 forward-backward，返回每个可见高斯的（屏幕梯度范数、高斯id、屏幕xy、可见mask）。"""
    from gsplat import rasterization
    for v in splats.values():
        if v.grad is not None:
            v.grad = None
    colors = torch.cat([splats["sh0"], splats["shN"]], 1)
    renders, alphas, info = rasterization(
        means=splats["means"], quats=splats["quats"],
        scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]),
        colors=colors, viewmats=torch.linalg.inv(c2w), Ks=K,
        width=w, height=h, sh_degree=sh_degree, absgrad=True, packed=False)
    pred = renders[0].clamp(0, 1)
    diff = (pred - gt).abs().mean(dim=2)
    if Wmap is None:
        loss = 0.8 * diff.mean()
    else:
        loss = 0.8 * (Wmap * diff).sum() / Wmap.sum()
    key = info["means2d"]
    key.retain_grad()
    loss.backward()
    g = key.absgrad.clone()
    if g is None:
        g = key.grad.clone()
    g[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
    g[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
    gn = g.norm(dim=-1)[0]          # [N]
    xy = key.detach()[0, :, :2]     # [N,2] 像素
    sel = (info["radii"][0] > 0).all(dim=-1)
    return gn, xy, sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(DATA_ROOT, "fieldH"))
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--max_images", type=int, default=331)
    ap.add_argument("--sh_degree", type=int, default=2)
    ap.add_argument("--root", default=os.path.join(OUT_ROOT, "v4_fieldH"))
    ap.add_argument("--model", default="base")
    ap.add_argument("--ckpt_step", type=int, default=10000)
    ap.add_argument("--h_radius_m", type=float, default=6.0)
    ap.add_argument("--weight", type=float, default=3.0)
    ap.add_argument("--tag", default="_holdout")
    args = ap.parse_args()
    device = "cuda"

    ckpt = os.path.join(args.root, f"seed0_{args.model}", f"ckpt_{args.ckpt_step}.ply")
    if not os.path.exists(ckpt):
        print("无 ckpt，回退最终模型：", ckpt)
        ckpt = os.path.join(args.root, f"seed0_{args.model}", "mini_trained.ply")
    print("加载", ckpt)

    data = load_scene(args.data_dir, args.factor, args.max_images)
    n = len(data["pixels"])
    val_ids = list(range(7, n, 8))
    H3, scale, alt = estimate_h3d(data, args.data_dir)
    c2w_all = data["camtoworlds"]
    splats = load_ply_splats(ckpt, args.sh_degree, device)
    for v in splats.values():
        v.requires_grad_(True)   # 从 PLY 加载的是普通张量，需开启梯度以取 means2d.absgrad

    rows = []
    agg_std, agg_w, cnt_std, cnt_w = {}, {}, {}, {}
    for idx in val_ids:
        gt = data["pixels"][idx].to(device)
        w, h = data["wh"][idx]
        c2w = c2w_all[idx:idx + 1].to(device)
        K = data["Ks"][idx:idx + 1].to(device)
        viewmat = torch.linalg.inv(c2w)[0].detach().cpu().numpy()
        pc = viewmat @ np.array([H3[0], H3[1], H3[2], 1.0])
        if pc[2] <= 0:
            continue
        Kn = K[0].cpu().numpy()
        u = Kn[0, 0] * pc[0] / pc[2] + Kn[0, 2]
        v = Kn[1, 1] * pc[1] / pc[2] + Kn[1, 2]
        rpx = Kn[0, 0] * (args.h_radius_m / scale) / pc[2]
        if not (0 <= u <= w and 0 <= v <= h) or rpx < 2:
            continue

        # 标准 L1
        splats["means"].grad = None
        gn0, xy0, sel0 = screen_grad(splats, c2w, K, w, h, gt, args.sh_degree, None)
        d2 = (xy0[:, 0] - u) ** 2 + (xy0[:, 1] - v) ** 2
        inH = sel0 & (d2 <= rpx ** 2)
        gH_std = gn0[inH].mean().item() if inH.any() else np.nan
        gO_std = gn0[sel0 & ~inH].mean().item()
        nH = int(inH.sum().item())
        # 多视角聚合（sum 后除以 count）
        for gi in torch.where(inH)[0].tolist():
            agg_std[gi] = agg_std.get(gi, 0.0) + gn0[gi].item(); cnt_std[gi] = cnt_std.get(gi, 0) + 1

        # ROI 加权 L1
        splats["means"].grad = None
        Wmap = roi_weight_map(c2w, K, w, h, H3, scale, args.h_radius_m,
                              args.weight, device, roi_mode="3d")
        gn1, xy1, sel1 = screen_grad(splats, c2w, K, w, h, gt, args.sh_degree, Wmap)
        d2w = (xy1[:, 0] - u) ** 2 + (xy1[:, 1] - v) ** 2
        inHw = sel1 & (d2w <= rpx ** 2)
        gH_w = gn1[inHw].mean().item() if inHw.any() else np.nan
        gO_w = gn1[sel1 & ~inHw].mean().item()
        for gi in torch.where(inHw)[0].tolist():
            agg_w[gi] = agg_w.get(gi, 0.0) + gn1[gi].item(); cnt_w[gi] = cnt_w.get(gi, 0) + 1

        N = float(w * h)
        p_gauss = 2.0 * np.pi * rpx ** 2 / N
        ratio = gH_w / max(gH_std, 1e-12)
        rows.append(dict(alt=float(alt[idx]), nH=nH, rpx=float(rpx),
                         p_gauss=float(p_gauss),
                         gH_std=gH_std, gH_w=gH_w, gO_std=gO_std, gO_w=gO_w,
                         ratio_h=ratio,
                         theory=args.weight / (1 + (args.weight - 1) * p_gauss)))
        del gn0, gn1, xy0, xy1

    rows.sort(key=lambda r: r["alt"])
    cpath = os.path.join(OUTD, f"v4_grad_probe{args.tag}.csv")
    with open(cpath, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)

    # 多视角聚合比值（H 区，按训练中 grad2d=sum/count 口径）
    common = set(agg_std) & set(agg_w)
    mv_std = np.mean([agg_std[i] / cnt_std[i] for i in common])
    mv_w = np.mean([agg_w[i] / cnt_w[i] for i in common])
    mv_ratio = mv_w / max(mv_std, 1e-12)

    print("\n===== 任务62：H 区平均屏幕梯度随高度（base 标准 L1）=====")
    print(f"{'高度':>8}{'nH':>6}{'gH_std':>12}{'阈值×':>9}")
    for r in rows:
        print(f"{r['alt']:>8.1f}{r['nH']:>6}{r['gH_std']:>12.6f}{r['gH_std']/GROW_GRAD2D:>9.2f}")
    below = [r for r in rows if r["gH_std"] < GROW_GRAD2D]
    if below:
        print(f"标准致密化 H 区屏幕梯度低于阈值 τ={GROW_GRAD2D} 的最高高度："
              f"{min(r['alt'] for r in below):.1f} m 起失效（共 {len(below)} 个视角）")

    print("\n===== 任务64：单视角加权/标准 H 区梯度比值 =====")
    for r in rows:
        print(f"高度{r['alt']:>7.1f}m  实测比值={r['ratio_h']:.2f}  纯归一化理论={r['theory']:.2f}")
    print(f"\n多视角聚合（grad2d=sum/count）加权/标准比值 = {mv_ratio:.2f}")
    print(f"纯归一化理论中位值 = {np.median([r['theory'] for r in rows]):.2f}")
    print(f"→ 单视角实测 {np.median([r['ratio_h'] for r in rows]):.2f} 接近理论，"
          f"多视角聚合 {mv_ratio:.2f} 被稀释，对照旧 grad_trace 实测 1.19~1.29")

    # 图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
        if os.path.exists(fp):
            font_manager.fontManager.addfont(fp)
            matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
            break
    matplotlib.rcParams["axes.unicode_minus"] = False
    alts = [r["alt"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].semilogy(alts, [r["gH_std"] for r in rows], "o-", color="#c0392b", label="H 区实测屏幕梯度")
    ax[0].axhline(GROW_GRAD2D, color="k", ls="--", lw=1.5, label="致密化阈值 τ=2e-4")
    ax[0].set_xlabel("飞行高度 / m"); ax[0].set_ylabel("H 区平均屏幕梯度 |∂L/∂μ₂D|")
    ax[0].set_title("任务62：小目标屏幕梯度随高度衰减（base）"); ax[0].grid(alpha=0.3); ax[0].legend()
    ax[1].plot(alts, [r["ratio_h"] for r in rows], "o-", color="#c0392b", label="单视角实测比值")
    ax[1].plot(alts, [r["theory"] for r in rows], "s--", color="#2471a3", label="纯归一化理论")
    ax[1].axhline(mv_ratio, color="#7d3c98", ls=":", label=f"多视角聚合={mv_ratio:.2f}")
    ax[1].set_xlabel("飞行高度 / m"); ax[1].set_ylabel("加权/标准 H 区梯度比值")
    ax[1].set_title("任务64：归一化(理论) vs 多视角平均(实测)"); ax[1].grid(alpha=0.3); ax[1].legend()
    fig.tight_layout()
    figp = os.path.join(OUTD, f"v4_grad_probe{args.tag}.png")
    plt.savefig(figp, dpi=150, bbox_inches="tight")
    print("\nSAVED", cpath)
    print("SAVED", figp)
    print("GRAD_PROBE_DONE")


if __name__ == "__main__":
    main()
