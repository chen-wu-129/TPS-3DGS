# --- open-source path shim (added for release) ---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import REPO_ROOT, DATA_ROOT, OUT_ROOT, RESULTS_ROOT  # noqa: E402
from imgio import imread_any  # noqa: E402
# -------------------------------------------------
# -*- coding: utf-8 -*-
"""roi_adaptive_fig.py —— ROI 权重区随高度自适应示意图（v4 健壮版）。

在已渲染的合成图上，按 H 的 4 m 标签框反推 6 m 物理半径 ROI 的屏幕圆，
叠加红色高斯权重蒙版（中心权重高、边缘衰减），直观展示：
三维固定 6 m 物理半径 → 投影后高空圆小、低空圆大，同一套参数无需逐高度调。

默认使用训练集 ours（seed=0，20x16=320 机位，高度层密、候选多）；ROI 圆几何
只由 H3 投影决定、与渲染配置无关，故不涉及方法优劣。纯 cv2/numpy/matplotlib，不占 GPU。

用法：
  python roi_adaptive_fig.py                 # 默认 tag=ours
  python roi_adaptive_fig.py --tag test_ours # 用 held-out 测试机位
"""
import os
import csv
import argparse
import numpy as np
import cv2
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

ROOT = os.path.join(REPO_ROOT, "yolo_syn")
W, H = 2048, 1152
ROI_M, LABEL_M = 6.0, 4.0     # ROI 物理半径 6 m；标签框为 4 m 地面正方形


def read_label(lbl_dir, idx):
    p = os.path.join(lbl_dir, f"syn_{idx:04d}.txt")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        t = f.read().split()
    if len(t) < 5:
        return None
    return [float(v) for v in t[1:5]]      # cx, cy, bw, bh（归一化）


def pick_one(rows, img_dir, lbl_dir, target, band=8.0):
    """在目标高度 ±band 内，优先选接近垂直俯视（H 框近正方形、水平半径小）
    且 H 居中的视角，使屏幕圆大小严格随高度反比、不被斜视透视压缩污染。"""
    cand = []
    for r in rows:
        if r["has_H"] != 1 or not (target - band <= r["alt"] <= target + band):
            continue
        lab = read_label(lbl_dir, r["idx"])
        if lab is None:
            continue
        cx, cy, bw, bh = lab
        if not (0.008 < bw < 0.95 and 0.008 < bh < 0.95 and 0.03 < cx < 0.97 and 0.03 < cy < 0.97):
            continue
        if not os.path.exists(os.path.join(img_dir, f"syn_{r['idx']:04d}.jpg")):
            continue
        aspect = abs(bw / max(bh, 1e-6) - 1.0)           # 0=框为正方形（俯视）
        center_dist = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
        cand.append((aspect, r.get("r", 0.0), center_dist, r["idx"], r["alt"], lab))
    if not cand:
        return None
    # 先按长宽比分桶（0–0.15 最优、0.15–0.30 次之…），桶内再按水平半径小、居中排序
    cand.sort(key=lambda c: (round(c[0] / 0.15), c[1], c[2]))
    return cand[0]


def overlay(img_dir, idx, lab):
    im = imread_any(os.path.join(img_dir, f"syn_{idx:04d}.jpg"))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    cx, cy, bw, bh = lab
    u, v = cx * W, cy * H
    rpx = 1.5 * bw * W              # 4 m 框半宽(bw*W/2)=2 m；6 m 圆半径 = 3×2 m
    yy, xx = np.mgrid[0:H, 0:W]
    g = np.exp(-((xx - u) ** 2 + (yy - v) ** 2) / (2.0 * max(rpx, 1) ** 2))
    alpha = 0.40 * g               # 红色权重蒙版
    red = np.zeros_like(im)
    red[..., 0] = 230
    red[..., 1] = 60
    red[..., 2] = 40
    im = (im * (1 - alpha[..., None]) + red * alpha[..., None]).astype(np.uint8)
    cv2.circle(im, (int(u), int(v)), int(rpx), (230, 60, 40), 3)   # 6 m 圆边界（红，与权重区同色）
    x0, y0 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)          # 4 m 标签框（绿）
    x1, y1 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
    cv2.rectangle(im, (x0, y0), (x1, y1), (60, 220, 120), 3)
    cv2.circle(im, (int(u), int(v)), 6, (255, 255, 255), -1)
    return im, rpx, u, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ours", help="images/labels 子目录名，如 ours / test_ours")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "budget_fig/roi_adaptive.png"))
    ap.add_argument("--targets", type=float, nargs=3, default=[30.0, 63.0, 105.0])
    args = ap.parse_args()

    img_dir = os.path.join(ROOT, "images", args.tag)
    lbl_dir = os.path.join(ROOT, "labels", args.tag)
    views_csv = os.path.join(ROOT, f"views_{args.tag}.csv")
    for p in (img_dir, lbl_dir, views_csv):
        if not os.path.exists(p):
            raise SystemExit(f"缺少输入：{p}（先渲染对应 tag）")

    with open(views_csv, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["alt"] = float(r["alt"])
        r["has_H"] = int(r["has_H"])
        r["idx"] = int(r["idx"])
        r["r"] = float(r.get("r", 0.0) or 0.0)

    picks = []
    for t in args.targets:
        p = pick_one(rows, img_dir, lbl_dir, t, band=8.0)
        if p is None:
            p = pick_one(rows, img_dir, lbl_dir, t, band=18.0)
        if p is not None:
            picks.append(p)
    if not picks:
        raise SystemExit("未选到任何 H 完整居中的视角，检查图像/标签/views 一致性。")

    fig, axs = plt.subplots(1, len(picks), figsize=(6.2 * len(picks), 4.0))
    if len(picks) == 1:
        axs = [axs]
    for k, (ax, (_, _, _, idx, alt, lab)) in enumerate(zip(axs, picks)):
        im, rpx, u, v = overlay(img_dir, idx, lab)
        ax.imshow(im)
        ax.set_title(f"飞行高度 {alt:.0f} m ｜ ROI 屏幕半径 {rpx:.0f} px", fontsize=13)
        ax.axis("off")
        if k >= 1:   # 中高空 ROI 较小，加局部放大插图
            half = 1.7 * rpx
            xc0, xc1 = max(int(u - half), 0), min(int(u + half), W)
            yc0, yc1 = max(int(v - half), 0), min(int(v + half), H)
            axins = ax.inset_axes([0.58, 0.10, 0.40, 0.44])
            axins.imshow(im)
            axins.set_xlim(xc0, xc1)
            axins.set_ylim(yc1, yc0)
            axins.set_xticks([])
            axins.set_yticks([])
            ax.indicate_inset_zoom(axins, edgecolor="black", lw=1.4)
    fig.suptitle("高度自适应的 ROI 权重区：三维固定 6 m 半径，投影后自动随高度缩放"
                 "（红=加权损失重点区，绿=H 标签框）", fontsize=13.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print("picked idx/alt/box_w_px:", [(p[3], round(p[4], 1), round(p[5][2] * W)) for p in picks])
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
