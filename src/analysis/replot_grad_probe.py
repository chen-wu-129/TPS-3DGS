# --- open-source path shim (added for release) ---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import REPO_ROOT, DATA_ROOT, OUT_ROOT, RESULTS_ROOT  # noqa: E402
# -------------------------------------------------
# -*- coding: utf-8 -*-
"""replot_grad_probe.py —— 读 ck5k/ck10k 两个时刻探针 CSV，纯 CPU 重绘合并图（mathtext）。"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUTD = RESULTS_ROOT
TAU = 2e-4
for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"


def main():
    """读探针 CSV 并重绘合并图；CSV 需先由 grad_probe_v4.py 生成。"""

    def load(tag):
        rows = []
        with open(os.path.join(OUTD, f"v4_grad_probe{tag}.csv"), encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                rows.append({k: float(v) for k, v in r.items()})
        rows.sort(key=lambda r: r["alt"])
        return rows


    r10 = load("_holdout")
    r5 = load("_holdout_ck5k")
    mv = 2.37

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.7))
    ax[0].semilogy([r["alt"] for r in r5], [r["gH_std"] for r in r5], "o-",
                   color="#e67e22", label="早期 ckpt-5000（致密化窗口内）")
    ax[0].semilogy([r["alt"] for r in r10], [r["gH_std"] for r in r10], "s-",
                   color="#c0392b", label="后期 ckpt-10000")
    ax[0].axhline(TAU, color="k", ls="--", lw=1.6, label=r"致密化阈值 $\tau=2\times10^{-4}$")
    ax[0].set_xlabel("飞行高度 / m")
    ax[0].set_ylabel(r"H 区平均屏幕梯度 $|\partial L/\partial \mu_{2D}|$")
    ax[0].set_title("小目标屏幕梯度在致密化窗口内全高度低于阈值")
    ax[0].grid(alpha=0.3, which="both"); ax[0].legend(fontsize=9)
    ax[0].annotate("最近 27.6 m 早期峰值仅 0.33$\\tau$", xy=(27.6, 6.7e-5),
                   xytext=(45, 1.2e-4), fontsize=9,
                   arrowprops=dict(arrowstyle="->", color="gray"))

    ax[1].plot([r["alt"] for r in r10], [r["ratio_h"] for r in r10], "o-",
               color="#c0392b", label="单视角实测（加权/标准）")
    ax[1].plot([r["alt"] for r in r10], [r["theory"] for r in r10], "s--",
               color="#2471a3", label=r"纯归一化理论 $w/[1+(w-1)p]$")
    ax[1].axhline(mv, color="#7d3c98", ls=":", lw=1.8, label=f"多视角聚合 $\\approx {mv}$")
    ax[1].set_xlabel("飞行高度 / m")
    ax[1].set_ylabel("加权/标准 H 区梯度比值")
    ax[1].set_title("损失加权对 H 区梯度的有效放大（非区内外比）")
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=9)
    fig.tight_layout()
    figp = os.path.join(OUTD, "v4_grad_probe_holdout.png")
    plt.savefig(figp, dpi=150, bbox_inches="tight")
    print("SAVED", figp)

    # 过阈值所需放大倍数诊断
    print("\n=== 纯损失加权(w=3, 实测~2.5x)能否过阈值（ckpt-5000 早期）===")
    for r in r5:
        need = TAU / max(r["gH_std"], 1e-12)
        after = r["gH_std"] * r["ratio_h"]
        flag = "过阈" if after > TAU else "未过"
        if r["alt"] in (27.6, 36.9, 42.3, 49.5, 59.8, 80.1, 98.1, 118.1):
            print(f"高度{r['alt']:>6.1f}m 基线={r['gH_std']/TAU:.2f}τ  需×{need:.1f}  "
                  f"加权后={after/TAU:.2f}τ [{flag}]")


if __name__ == "__main__":
    main()
