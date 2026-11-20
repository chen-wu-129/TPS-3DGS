# --- open-source path shim (added for release) ---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import REPO_ROOT, DATA_ROOT, OUT_ROOT, RESULTS_ROOT  # noqa: E402
# -------------------------------------------------
# -*- coding: utf-8 -*-
"""budget_stats_v4.py —— 21 个 v4 模型 PLY 预算重分配统计（7 配置 x 3 seeds）

直接解析 PLY（不依赖 torch/gsplat），输出：
  v4_ply_stats.csv              每模型：总数/ROI数/占比/密度/尺度/不透明度
  v4_ply_stats_agg.csv          每配置 3 seeds 均值±SD
  budget_fig/v4_radial_density.png  径向地表密度曲线（7 配置）
"""
import os, csv, statistics
import numpy as np
import plyfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# H3 必须与训练/评测动态估计值（train_ours.estimate_h3d：camtoworlds 水平均值 + 局部点云地面主峰）完全一致。
# 权威 H3=[0.06973587 0.04483227 0.00587415]（见 h3_util.py；旧 -0.06562055 为全局5%分位，悬空约7m，已废弃）。
H3 = np.array([0.06973587, 0.04483227, 0.00587415])
SCALE = 99.433
H_RADIUS_M = 6.0
Z_BAND_M = 3.0
RINGS = [0, 2, 4, 6, 8, 10, 14, 20, 30]

ROOT = os.path.join(OUT_ROOT, "v4_fieldH")
OUTD = RESULTS_ROOT
FIGD = os.path.join(REPO_ROOT, "budget_fig")
os.makedirs(FIGD, exist_ok=True)

CONFIGS = [
    ("base", "标准3DGS", "#5b8db8"),
    ("ours", "本文方法(w=3)", "#c0392b"),
    ("w5", "权重w=5", "#2e8b57"),
    ("focal", "Focal loss", "#8e8e8e"),
    ("randroi", "随机ROI加权", "#b8860b"),
    ("gradscale", "梯度缩放", "#6a5acd"),
    ("budget", "预算x1.2", "#20b2aa"),
]
SEEDS = [0, 1, 2]


def load_geom(path):
    v = plyfile.PlyData.read(path)["vertex"]
    names = v.data.dtype.names
    x = v["x"].astype(np.float64)
    y = v["y"].astype(np.float64)
    z = v["z"].astype(np.float64)
    op = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))
    scn = sorted(k for k in names if k.startswith("scale_"))
    scales = np.stack([v[k].astype(np.float64) for k in scn], axis=1)
    smax = np.exp(np.max(scales, axis=1)) * SCALE
    return x, y, z, op, smax


def stats_for(path):
    x, y, z, op, smax = load_geom(path)
    dh = np.sqrt((x - H3[0]) ** 2 + (y - H3[1]) ** 2) * SCALE
    z_g = float(np.median(z[dh < H_RADIUS_M]))
    zg = np.abs(z - z_g) * SCALE
    ground = zg < Z_BAND_M
    in_h = (dh < H_RADIUS_M) & ground
    out_h = (~in_h)
    total = len(x)
    n_h = int(in_h.sum())
    # 不透明度加权的 ROI 密度（可见性代理：op 越高贡献越大）
    opw_density = float(op[in_h].sum() / (np.pi * H_RADIUS_M ** 2))
    dens = []
    for i in range(len(RINGS) - 1):
        a, b = RINGS[i], RINGS[i + 1]
        m = (dh >= a) & (dh < b) & ground
        dens.append(float(m.sum() / (np.pi * (b ** 2 - a ** 2))))
    return {
        "n_total": total,
        "n_roi": n_h,
        "roi_pct": 100.0 * n_h / total,
        "roi_density": n_h / (np.pi * H_RADIUS_M ** 2),
        "roi_opw_density": opw_density,
        "roi_size": float(smax[in_h].mean()),
        "roi_opacity": float(op[in_h].mean()),
        "bg_size": float(smax[out_h].mean()),
        "bg_opacity": float(op[out_h].mean()),
        "rings": dens,
    }


EXTRA_CONFIGS = [
    ("w2", "权重w=2", "#e67e22"),
    ("w4", "权重w=4", "#16a085"),
    ("r3", "半径r=3m", "#7f8c8d"),
    ("r9", "半径r=9m", "#c71585"),
]

DECOMPOSE_CONFIGS = [
    ("base", "标准3DGS", "#5b8db8"),
    ("lossonly", "仅ROI损失加权", "#e67e22"),
    ("salonly", "仅致密化梯度重标定", "#16a085"),
    ("ours", "本文(叠加,w=3)", "#c0392b"),
    ("w5", "叠加(w=5)", "#2e8b57"),
    ("focal", "Focal loss", "#8e8e8e"),
    ("randroi", "随机ROI", "#b8860b"),
    ("randfix", "固定错误位置ROI", "#d63384"),
    ("gradscale", "梯度缩放", "#6a5acd"),
    ("budget", "预算x1.2", "#20b2aa"),
]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--extra", action="store_true", help="加入 w2/w4/r3/r9 消融配置")
    ap.add_argument("--decompose", action="store_true",
                    help="2x2 机制拆解矩阵：base/lossonly/salonly/ours/w5/...")
    args = ap.parse_args()
    root = args.root
    seeds = args.seeds
    if args.decompose:
        configs = DECOMPOSE_CONFIGS
    else:
        configs = CONFIGS + (EXTRA_CONFIGS if args.extra else [])
    per = {}
    detail = []
    for cfg, lab, col in configs:
        vals = []
        for s in seeds:
            p = os.path.join(root, f"seed{s}_{cfg}", "mini_trained.ply")
            if not os.path.exists(p):
                print("MISSING", p); continue
            d = stats_for(p)
            vals.append(d)
            detail.append((cfg, s, d))
        per[cfg] = vals

    cols = ["n_total", "n_roi", "roi_pct", "roi_density", "roi_opw_density",
            "roi_size", "roi_opacity", "bg_size", "bg_opacity"]
    cpath = os.path.join(OUTD, f"v4_ply_stats{args.tag}.csv")
    with open(cpath, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["config", "seed"] + cols)
        for cfg, s, d in detail:
            w.writerow([cfg, s] + [f"{d[c]:.4f}" for c in cols])

    apath = os.path.join(OUTD, f"v4_ply_stats_agg{args.tag}.csv")
    with open(apath, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["config", "label"] + cols)
        for cfg, lab, col in configs:
            vals = per[cfg]
            row = [cfg, lab]
            for c in cols:
                xs = [v[c] for v in vals]
                if not xs:
                    row.append("N/A"); continue
                m = statistics.mean(xs)
                sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
                row.append(f"{m:.4f}±{sd:.4f}")
            w.writerow(row)

    print(f"{'config':<10}{'N_total':>14}{'N_ROI':>9}{'ROI%':>8}{'密度/m2':>9}{'不透明度':>9}")
    for cfg, lab, col in configs:
        vals = per[cfg]
        for i, v in enumerate(vals):
            print(f"{cfg}{i:<7}{v['n_total']:>14,.0f}{v['n_roi']:>9,.0f}{v['roi_pct']:>8.3f}"
                  f"{v['roi_density']:>9.2f}{v['roi_opacity']:>9.3f}")
    print()
    for cfg, lab, col in configs:
        vals = per[cfg]
        if not vals:
            print(f"{lab:<14} N/A (模型缺失，跳过)"); continue
        ms = lambda c: (statistics.mean([v[c] for v in vals]),
                        statistics.stdev([v[c] for v in vals]) if len(vals) > 1 else 0)
        nt, nts = ms("n_total"); nr, nrs = ms("n_roi"); rp, rps = ms("roi_pct")
        de, des = ms("roi_density"); opm, ops = ms("roi_opacity")
        print(f"{lab:<14} N={nt:,.0f}±{nts:,.0f}  ROI={nr:,.0f}±{nrs:,.0f}  "
              f"{rp:.3f}±{rps:.3f}%  密度={de:.1f}±{des:.1f}  不透明度={opm:.3f}±{ops:.3f}")

    centers = [(RINGS[i] + RINGS[i + 1]) / 2 for i in range(len(RINGS) - 1)]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for cfg, lab, col in configs:
        vals = per[cfg]
        if not vals:
            continue
        mean_rings = [np.mean([v["rings"][i] for v in vals]) for i in range(len(RINGS) - 1)]
        lw = 2.4 if cfg in ("base", "lossonly", "salonly", "ours") else 1.3
        ls = "-" if cfg in ("base", "lossonly", "salonly", "ours", "w5") else "--"
        ax.plot(centers, mean_rings, ls + "o", color=col, label=lab, lw=lw, ms=4)
    ax.axvspan(0, H_RADIUS_M, color="#c0392b", alpha=0.08)
    ax.text(H_RADIUS_M / 2, ax.get_ylim()[1] * 0.97, "H 着陆区(半径6 m)",
            ha="center", va="top", color="#c0392b", fontsize=10)
    ax.set_xlabel("到 H 中心的水平距离 / m")
    ax.set_ylabel("地表高斯密度 / 个·m$^{-2}$")
    ax.set_title("各方法地表高斯密度的径向分布")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    png = os.path.join(FIGD, f"v4_radial_density{args.tag}.png")
    fig.savefig(png, dpi=180)
    print("\nSAVED", cpath)
    print("SAVED", apath)
    print("SAVED", png)


if __name__ == "__main__":
    main()
