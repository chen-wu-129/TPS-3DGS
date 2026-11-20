# --- open-source path shim (added for release) ---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import REPO_ROOT, DATA_ROOT, OUT_ROOT, RESULTS_ROOT  # noqa: E402
# -------------------------------------------------
# -*- coding: utf-8 -*-
"""图 1 终版：按最终印刷尺寸设计（6.0 x 4.35 in ≈ 15.2 x 11.0 cm），文字不经缩放。
红框坐标取自作者 PPT 的 EMU 换算。"""
import csv
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

sys.path.insert(0, REPO_ROOT)
from eval_altitude import load_ply_splats, render_view
from h3_util import correct_h3
from mini_train import load_scene
from read_altitude import read_xmp

plt.rcParams.update({"font.sans-serif": ["SimHei"], "axes.unicode_minus": False,
                     "font.size": 6.2, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2.0, "ytick.major.size": 2.0})

ROOT = os.path.join(OUT_ROOT, "v4_fieldH")
DATA = os.path.join(DATA_ROOT, "fieldH")
PERVIEW = os.path.join(OUT_ROOT, "v5/diag_budget_ea/per_view_real_altitude.csv")
YOLO = os.path.join(RESULTS_ROOT, "yolo闭环")
OUT = os.path.join(OUT_ROOT, "figures", "fig1_problem_final.png")
ALTS = [42, 60, 80, 118]
BINS = [(25, 45), (45, 60), (60, 85), (85, 115)]
BLAB = ["25–45", "45–60", "60–85", "85–115"]

BOXES = [
    (0.10398, 0.50195, 0.06041, 0.04528),
    (0.36437, 0.50195, 0.06041, 0.04528),
    (0.6030, 0.4811, 0.06041, 0.04528),
    (0.84905, 0.57620, 0.06041, 0.04528),
]
LABELS = [
    (0.13420, "白色画笔被染色"),
    (0.39458, "白色画笔被染色"),
    (0.62934, "白高斯点被抑制"),
    (0.87926, "笔画被压暗，边缘出现暗晕"),
]
LABEL_Y = 0.6755
RED = "#d00000"


def pct_fmt(v):
    if abs(v - round(v)) < 1e-6:
        return "%d%%" % round(v)
    return "%.1f%%" % v


def sat(x):
    mx = x.max(axis=2)
    mn = x.min(axis=2)
    return np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)


def main():
    data = load_scene(DATA, 4, 331)
    n = len(data["pixels"])
    val_ids = list(range(7, n, 8))
    c2w_all = data["camtoworlds"]
    cam = c2w_all[:, :3, 3].numpy()
    pts = data["points"].numpy()
    alt = []
    for nm in data["names"]:
        rel, _, _ = read_xmp(os.path.join(DATA, "images", nm))
        alt.append(rel if rel is not None else np.nan)
    alt = np.array(alt)
    m = ~np.isnan(alt)
    A = np.polyfit(cam[m, 2], alt[m], 1)
    scale = abs(A[0])
    H3 = correct_h3(cam, pts, hax=(0, 1), b_axis=2)

    def project(i):
        c2w = c2w_all[i:i + 1].to("cuda")
        K = data["Ks"][i:i + 1].to("cuda")
        vm = torch.linalg.inv(c2w)[0].cpu().numpy()
        pc = vm @ np.array([H3[0], H3[1], H3[2], 1.0])
        Kn = K[0].cpu().numpy()
        w, h = data["wh"][i]
        if pc[2] <= 0:
            return None
        q = Kn @ pc[:3]
        u, v = q[0] / q[2], q[1] / q[2]
        rpx = Kn[0, 0] * (4.0 / scale) / pc[2]
        if not (0 <= u <= w and 0 <= v <= h) or rpx < 2:
            return None
        return c2w, K, w, h, u, v, rpx

    valid = [i for i in val_ids if project(i) is not None]
    picks = []
    for t in ALTS:
        i = min(valid, key=lambda j: abs(alt[j] - t))
        if i not in picks:
            picks.append(i)

    s = load_ply_splats(os.path.join(ROOT, "seed0_budget", "mini_trained.ply"), 2, "cuda")
    tiles = []
    for i in picks:
        c2w, K, w, h, u, v, rpx = project(i)
        with torch.no_grad():
            pr, _ = render_view(s, c2w, K, w, h, 2)
        pr = pr.cpu().numpy()
        gt = np.clip(data["pixels"][i].cpu().numpy(), 0, 1) * 255
        pb = np.clip(pr, 0, 1) * 255
        r = int(1.55 * rpx) + 2
        x0, x1 = int(max(0, u - r)), int(min(w, u + r))
        y0, y1 = int(max(0, v - r)), int(min(h, v + r))
        cg, cp = gt[y0:y1, x0:x1], pb[y0:y1, x0:x1]
        tiles.append((alt[i], rpx, cg, cp, sat(cg).mean(), sat(cp).mean()))
    del s

    fig = plt.figure(figsize=(6.0, 4.35), dpi=600)
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 1.30],
                  left=0.052, right=0.985, top=0.955, bottom=0.075,
                  wspace=0.16, hspace=0.28)

    for ci, (a, rpx, cg, cp, sg, sp) in enumerate(tiles):
        ax = fig.add_subplot(gs[0, ci])
        ax.imshow(cv2.resize(cg.astype(np.uint8), (300, 300), interpolation=cv2.INTER_CUBIC))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("%.0f m" % a, fontsize=6.8, pad=1.5)
        if ci == 0:
            ax.set_ylabel("真值 GT", fontsize=6.8, labelpad=2)

        ax = fig.add_subplot(gs[1, ci])
        ax.imshow(cv2.resize(cp.astype(np.uint8), (300, 300), interpolation=cv2.INTER_CUBIC))
        ax.set_xticks([]); ax.set_yticks([])
        if ci == 0:
            ax.set_ylabel("标准 3DGS", fontsize=6.8, labelpad=2)
        ax.set_xlabel("饱和度 %.2f / 真值 %.2f" % (sp, sg), fontsize=5.6, labelpad=1)

    rows = [(float(x["alt_m"]), float(x["h_rpx"])) for x in
            csv.DictReader(open(PERVIEW, encoding="utf-8-sig")) if x["h_psnr"] not in ("nan", "")]
    rows.sort()
    sa = np.array([a for a, _ in rows]); sr = np.array([p for _, p in rows])
    axB = fig.add_subplot(gs[2, 0:2])
    axB.scatter(sa, sr, s=4.5, color="#c0392b", alpha=0.40, edgecolor="none")
    cx, cy = [], []
    for lo in range(20, 121, 10):
        mm = (sa >= lo) & (sa < lo + 10)
        if mm.sum():
            cx.append(sa[mm].mean()); cy.append(np.median(sr[mm]))
    axB.plot(cx, cy, "-o", color="#8e1f13", lw=1.2, ms=2.6, label="分段中位数")
    axB.set_xlabel("飞行高度 (m)", fontsize=6.4, labelpad=1)
    axB.set_ylabel("标志屏幕半径 (px)", fontsize=6.4)
    axB.set_title("(b) 标志在画面中单调收缩：20 → 7 px", fontsize=6.6, pad=2.5)
    axB.tick_params(labelsize=5.8)
    axB.grid(alpha=0.25, lw=0.35)
    axB.legend(fontsize=5.2, loc="upper right", framealpha=0.9, handlelength=1.2)

    x = np.arange(len(BINS))
    rr = {}
    for tag in ("real", "base"):
        rows2 = list(csv.DictReader(open(os.path.join(YOLO, "per_view_%s.csv" % tag), encoding="utf-8-sig")))
        v = []
        for lo, hi in BINS:
            sel = [q for q in rows2 if lo <= float(q["alt"]) < hi]
            v.append(100.0 * sum(1 for q in sel if q["det"].strip() == "1") / max(len(sel), 1))
        rr[tag] = v
    axC = fig.add_subplot(gs[2, 2:4])
    axC.plot(x, rr["real"], "-o", color="#1f5fa8", lw=1.2, ms=3.2, label="真实照片训练")
    axC.plot(x, rr["base"], "-s", color="#c0392b", lw=1.2, ms=3.2, label="标准 3DGS 合成图训练")
    for xi, v in zip(x, rr["real"]):
        axC.annotate(pct_fmt(v), (xi, v), textcoords="offset points", xytext=(0, 4.5),
                     ha="center", fontsize=5.2, color="#1f5fa8")
    for xi, v in zip(x, rr["base"]):
        axC.annotate(pct_fmt(v), (xi, v), textcoords="offset points", xytext=(0, -8),
                     ha="center", fontsize=5.2, color="#c0392b")
    axC.set_xticks(x); axC.set_xticklabels(BLAB, fontsize=5.6)
    axC.set_xlabel("真实照片测试的高度段 (m)", fontsize=6.4, labelpad=1)
    axC.set_ylabel("检出率 (%)", fontsize=6.4)
    axC.set_ylim(-12, 120)
    axC.set_title("(c) 下游检出率的断层", fontsize=6.6, pad=2.5)
    axC.tick_params(labelsize=5.8)
    axC.grid(alpha=0.25, lw=0.35, axis="y")
    axC.legend(fontsize=5.2, loc="center right", framealpha=0.92, handlelength=1.2)

    for (fx, fy, fw, fh) in BOXES:
        fig.add_artist(Rectangle((fx, fy), fw, fh, transform=fig.transFigure,
                                 fill=False, ec=RED, lw=1.0, zorder=50))
    for (lx, txt) in LABELS:
        fig.text(lx, LABEL_Y, txt, color=RED, fontsize=6.6, fontweight="bold",
                 ha="center", va="center", zorder=51)

    fig.text(0.5185, 0.982, "(a)", fontsize=7.0, fontweight="bold",
             ha="center", va="center", zorder=52)

    fig.savefig(OUT, facecolor="white")
    print("saved", OUT, "6.0x4.35 in @600dpi")


if __name__ == "__main__":
    main()