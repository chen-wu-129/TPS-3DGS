# --- open-source path shim (added for release) ---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import REPO_ROOT, DATA_ROOT, OUT_ROOT, RESULTS_ROOT  # noqa: E402
# -------------------------------------------------
# -*- coding: utf-8 -*-
"""
汇总物理多半径致密化事件计数（4 配置 _evt2，截至 step12000 致密化窗口末）。
输出：
  v4_event_counts_phys_holdout.csv   每配置各半径：窗口末高斯数、clone/split 累计
  v4_event_ring_density_holdout.csv  环带事件密度（事件/平方米）：区分笔画核心与外围地面
物理半径 1.5/2/3/4/6 m；环带 [0,1.5],[1.5,2],[2,3],[3,4],[4,6]。
"""
import csv, os, math
import numpy as np

BASE = os.path.join(OUT_ROOT, "v4_fieldH")
OUTDIR = RESULTS_ROOT
CFGS = ["base", "lossonly", "salonly", "ours"]
RADII = [1.5, 2.0, 3.0, 4.0, 6.0]
RING_EDGES = [0.0, 1.5, 2.0, 3.0, 4.0, 6.0]
STEP_MAX = 12000

# event_trace 列名
NCOL = ["n_r15", "n_r20", "n_r30", "n_r40"]          # n_r60 用 n_roi3d
CCOL = ["clo_r15", "clo_r20", "clo_r30", "clo_r40", "clo_r60"]
SCOL = ["spl_r15", "spl_r20", "spl_r30", "spl_r40", "spl_r60"]


def load(cfg):
    f = os.path.join(BASE, f"seed0_{cfg}_evt2", "event_trace.csv")
    rows = [r for r in csv.DictReader(open(f, encoding="utf-8"))
            if int(r["step"]) <= STEP_MAX]
    n_in = {1.5: [], 2.0: [], 3.0: [], 4.0: [], 6.0: []}
    clone = {r: 0 for r in RADII}
    split = {r: 0 for r in RADII}
    for r in rows:
        nvals = [int(r["n_r15"]), int(r["n_r20"]), int(r["n_r30"]),
                 int(r["n_r40"]), int(r["n_roi3d"])]
        for rad, nv in zip(RADII, nvals):
            n_in[rad].append(nv)
        for rad, c in zip(RADII, CCOL):
            clone[rad] += int(r[c])
        for rad, c in zip(RADII, SCOL):
            split[rad] += int(r[c])
    n_end = {rad: (n_in[rad][-1] if n_in[rad] else 0) for rad in RADII}
    n_mean = {rad: (float(np.mean(n_in[rad])) if n_in[rad] else 0.0) for rad in RADII}
    return n_end, n_mean, clone, split


def main():
    wide, rings = [], []
    for cfg in CFGS:
        n_end, n_mean, clone, split = load(cfg)
        row = {"config": cfg}
        for rad in RADII:
            tag = f"{rad:g}"
            row[f"n{tag}"] = n_end[rad]
            row[f"clone{tag}"] = clone[rad]
            row[f"split{tag}"] = split[rad]
        wide.append(row)
        # 环带（外半径累积值做差分）
        for i in range(1, len(RING_EDGES)):
            ro, ri = RING_EDGES[i], RING_EDGES[i - 1]
            area = math.pi * (ro ** 2 - ri ** 2)
            clo_o = clone[ro] if ro > 0 else 0
            clo_i = clone[ri] if ri > 0 else 0
            spl_o = split[ro] if ro > 0 else 0
            spl_i = split[ri] if ri > 0 else 0
            clo_ring = clo_o - clo_i
            spl_ring = spl_o - spl_i
            rings.append({
                "config": cfg,
                "ring_m": f"{ri:g}-{ro:g}",
                "area_m2": round(area, 2),
                "clone": clo_ring,
                "split": spl_ring,
                "clone_per_m2": round(clo_ring / area, 3),
                "split_per_m2": round(spl_ring / area, 3),
            })

    wpath = os.path.join(OUTDIR, "v4_event_counts_phys_holdout.csv")
    with open(wpath, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(wide[0].keys()))
        w.writeheader()
        w.writerows(wide)
    print("SAVED", wpath)
    for r in wide:
        print(r)

    rpath = os.path.join(OUTDIR, "v4_event_ring_density_holdout.csv")
    with open(rpath, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rings[0].keys()))
        w.writeheader()
        w.writerows(rings)
    print("SAVED", rpath)
    # 打印环带密度对比（clone_per_m2）
    print("\n=== clone 事件密度（个/m^2）===")
    hdr = ["config"] + [f"{r['ring_m']}m" for r in rings if r["config"] == CFGS[0]]
    print("\t".join(hdr))
    for cfg in CFGS:
        vals = [rr["clone_per_m2"] for rr in rings if rr["config"] == cfg]
        print(cfg + "\t" + "\t".join(f"{v:.2f}" for v in vals))


if __name__ == "__main__":
    main()
