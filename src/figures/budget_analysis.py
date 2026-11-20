# --- open-source path shim (added for release) ---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import REPO_ROOT, DATA_ROOT, OUT_ROOT, RESULTS_ROOT  # noqa: E402
# -------------------------------------------------
# -*- coding: utf-8 -*-
"""budget_analysis.py —— 固定高斯预算下，H 着陆区的高斯预算重分配定量分析。
纯 plyfile/numpy/matplotlib（不 import gsplat/torch），直接解析 gsplat 导出的 PLY。
输出：
  budget_fig/budget_stats.csv      三模型 H 区/背景 点数占比、密度、尺度、不透明度
  budget_fig/budget_radial_density.png  地表高斯密度随到 H 水平距离的径向曲线
"""
import os, csv
import numpy as np
import plyfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# H 三维坐标（归一化单位）、米/单位、竖直轴（来自 estimate_h3d，h3_refined.txt）
H3 = np.array([0.05416236, 0.04238823, -0.06562055])
SCALE = 99.433          # 1 归一化单位 = 99.433 m
VAX = 2                 # 竖直轴
H_RADIUS_M = 6.0        # H 着陆区半径
Z_BAND_M = 3.0          # 地表薄带（地面上下 3 m）
RINGS = [0, 2, 4, 6, 8, 10, 14, 20, 30]

MODELS = [
    ("Baseline 标准3DGS", os.path.join(OUT_ROOT, "fieldH/mini_trained.ply"), "#9aa5ab"),
    ("Ours-w3 本文(权重3)", os.path.join(OUT_ROOT, "fieldH_ours/mini_trained.ply"), "#c0392b"),
    ("Ours-w5 激进(权重5)", os.path.join(OUT_ROOT, "fieldH_ours_w5/mini_trained.ply"), "#8e44ad"),
]
OUT = os.path.join(REPO_ROOT, "budget_fig")
os.makedirs(OUT, exist_ok=True)


def load_geom(path):
    v = plyfile.PlyData.read(path)["vertex"]
    names = v.data.dtype.names
    x = v["x"].astype(np.float64)
    y = v["y"].astype(np.float64)
    z = v["z"].astype(np.float64)
    op = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))   # logit -> prob
    scn = sorted(k for k in names if k.startswith("scale_"))
    scales = np.stack([v[k].astype(np.float64) for k in scn], axis=1)
    smax = np.exp(np.max(scales, axis=1)) * SCALE                # 最大半宽，单位 m
    return x, y, z, op, smax


rows, ring = [], {}
for name, path, col in MODELS:
    x, y, z, op, smax = load_geom(path)
    dh = np.sqrt((x - H3[0]) ** 2 + (y - H3[1]) ** 2) * SCALE     # 水平距离(m)
    # 以各模型 H 水平圈内点的 z 中位数作为“当地地表”（最终 ply 优化后地表与
    # 初始 SfM 的 H3z 有数米偏差，不能用固定 H3z），再取上下 3m 地表薄带
    z_g = float(np.median(z[dh < H_RADIUS_M]))
    zg = np.abs(z - z_g) * SCALE                                  # 离当地地表高度(m)
    ground = zg < Z_BAND_M
    in_h = (dh < H_RADIUS_M) & ground
    total = len(x)
    rows.append({
        "model": name,
        "total": total,
        "H_count": int(in_h.sum()),
        "H_frac_pct": round(100.0 * in_h.sum() / total, 3),
        "H_density_per_m2": round(in_h.sum() / (np.pi * H_RADIUS_M ** 2), 3),
        "H_mean_size_m": round(float(smax[in_h].mean()), 4),
        "H_mean_opacity": round(float(op[in_h].mean()), 3),
        "bg_mean_size_m": round(float(smax[~in_h].mean()), 4),
        "bg_mean_opacity": round(float(op[~in_h].mean()), 3),
    })
    dens = []
    for i in range(len(RINGS) - 1):
        a, b = RINGS[i], RINGS[i + 1]
        m = (dh >= a) & (dh < b) & ground
        dens.append(m.sum() / (np.pi * (b ** 2 - a ** 2)))
    ring[name] = (dens, col)
    print(f"{name}: total={total:,} H={int(in_h.sum()):,} "
          f"({100*in_h.sum()/total:.2f}%) H密度={in_h.sum()/(np.pi*H_RADIUS_M**2):.2f}/m2 "
          f"H尺度={smax[in_h].mean():.3f}m 背景尺度={smax[~in_h].mean():.3f}m")

csv_path = os.path.join(OUT, "budget_stats.csv")
with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)

centers = [(RINGS[i] + RINGS[i + 1]) / 2 for i in range(len(RINGS) - 1)]
fig, ax = plt.subplots(figsize=(7.4, 4.4))
for name, (dens, col) in ring.items():
    ax.plot(centers, dens, "-o", color=col, label=name, lw=2, ms=5)
ax.axvspan(0, H_RADIUS_M, color="#c0392b", alpha=0.08)
ax.text(H_RADIUS_M / 2, ax.get_ylim()[1] * 0.97, "H 着陆区(半径6m)",
        ha="center", va="top", color="#c0392b", fontsize=10)
ax.set_xlabel("到 H 中心的水平距离 / m")
ax.set_ylabel("地表高斯密度 / 个·m$^{-2}$")
ax.set_title("固定总预算下本文方法把高斯密度向 H 着陆区集中")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
png = os.path.join(OUT, "budget_radial_density.png")
fig.savefig(png, dpi=180)
print("SAVED", csv_path)
print("SAVED", png)
