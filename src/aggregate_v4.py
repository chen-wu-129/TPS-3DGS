# --- open-source path shim (added for release) ---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import REPO_ROOT, DATA_ROOT, OUT_ROOT, RESULTS_ROOT  # noqa: E402
# -------------------------------------------------
# -*- coding: utf-8 -*-
"""aggregate_v4.py —— 聚合 9 个主对比模型（base/ours/w5 x 3 seeds）的事件计数与预算统计
输出 v4_budget_agg.csv：表2（预算重分配证据）数据
- 最终高斯总数 / ROI内高斯 / ROI占比
- 致密化窗口（step<=12000）内 ROI clone/split 事件累计数
- ROI 内外净增（总净增 - ROI净增）
"""
import csv, os, statistics

ROOT = os.path.join(OUT_ROOT, "v4_fieldH")
CONFIGS = ["base", "ours", "w5"]
SEEDS = [0, 1, 2]
REFINE_STOP = 12000


def load(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in r:
            r[k] = float(r[k])
    return rows


def stats(rows):
    first = rows[0]
    last = rows[-1]
    # 致密化停止窗口内（最后一个 step<=12000 的记录）
    win = [r for r in rows if r["step"] <= REFINE_STOP]
    wlast = win[-1] if win else first
    n_total0, n_total1 = first["n_gauss"], last["n_gauss"]
    n_roi0, n_roi1 = first["n_roi3d"], last["n_roi3d"]
    return {
        "n_total": n_total1,
        "n_roi": n_roi1,
        "roi_pct": 100.0 * n_roi1 / n_total1,
        "clone_roi_window": wlast["clone_roi"],     # 致密化窗口内累计
        "split_roi_window": wlast["split_roi"],
        "net_total": n_total1 - n_total0,
        "net_roi": n_roi1 - n_roi0,
        "net_out": (n_total1 - n_total0) - (n_roi1 - n_roi0),
        "opac_roi": last["opac_roi_mean"],
        "opac_out": last["opac_out_mean"],
    }


def main():
    per = {}
    for cfg in CONFIGS:
        vals = []
        for s in SEEDS:
            p = os.path.join(ROOT, f"seed{s}_{cfg}", "event_trace.csv")
            if not os.path.exists(p):
                print("MISSING", p); continue
            vals.append(stats(load(p)))
        per[cfg] = vals

    cols = ["n_total", "n_roi", "roi_pct", "clone_roi_window", "split_roi_window",
            "net_total", "net_roi", "net_out", "opac_roi", "opac_out"]
    out = os.path.join(REPO_ROOT, "v4_budget_agg.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["config", "seed"] + cols)
        for cfg in CONFIGS:
            for i, v in enumerate(per[cfg]):
                w.writerow([cfg, i] + [f"{v[c]:.4f}" for c in cols])
        # 均值±SD
        w.writerow([])
        w.writerow(["config", "mean±SD"] + cols)
        for cfg in CONFIGS:
            row = [cfg, "agg"]
            for c in cols:
                xs = [v[c] for v in per[cfg]]
                m = statistics.mean(xs)
                sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
                row.append(f"{m:.1f}±{sd:.1f}")
            w.writerow(row)

    # 控制台打印关键对比
    print(f"{'config':<8} {'N_total':>16} {'N_ROI':>12} {'ROI%':>7} {'clone_win':>10} {'split_win':>10} {'net_out':>12}")
    for cfg in CONFIGS:
        for i, v in enumerate(per[cfg]):
            print(f"{cfg}{i:<5} {v['n_total']:>16,.0f} {v['n_roi']:>12,.0f} {v['roi_pct']:>7.3f} "
                  f"{v['clone_roi_window']:>10,.0f} {v['split_roi_window']:>10,.0f} {v['net_out']:>12,.0f}")
    print()
    print(f"{'config':<8} {'N_total(mean±SD)':>20} {'N_ROI(mean±SD)':>18} {'ROI%':>8} {'clone_win':>10} {'split_win':>10}")
    for cfg in CONFIGS:
        def ms(c, d=0):
            xs = [v[c] for v in per[cfg]]
            return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)
        nt, nts = ms("n_total"); nr, nrs = ms("n_roi"); rp, _ = ms("roi_pct")
        cl, _ = ms("clone_roi_window"); sp, _ = ms("split_roi_window")
        print(f"{cfg:<8} {nt:>12,.0f}±{nts:<7,.0f} {nr:>10,.0f}±{nrs:<7,.0f} {rp:>8.3f} {cl:>10,.0f} {sp:>10,.0f}")
    print("\nWROTE", out)


if __name__ == "__main__":
    main()
