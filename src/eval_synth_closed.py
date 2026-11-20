# --- open-source path shim (added for release) ---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import REPO_ROOT, DATA_ROOT, OUT_ROOT, RESULTS_ROOT  # noqa: E402
# -------------------------------------------------
# -*- coding: utf-8 -*-
"""eval_synth_closed.py —— 纯合成闭环评测（v4，不再使用任何真实照片）

检测器（全部仅用合成图训练）在【held-out 合成测试机位】上考试：
  - base      : 标准 3DGS 合成训练
  - ours      : 本文方法(w=3) 合成训练
  - w5        : 权重消融 w=5
  - focal     : focal loss 基线
  - randroi   : 随机同面积 ROI 加权基线
  - gradscale : 梯度缩放基线
  - budget    : 预算 x1.2 基线
严格三分：训练 seed=0（20x16=320 张）、验证 seed=7（15x12=180，仅用于早停/选 best epoch）、
测试 seed=13（13x14=182，高度层 linspace(22,120,13) 与方位 n_az=14 与 val 均错开）；
best 模型仅在 val 上选出，test 全程不参与训练与模型选择，避免测试集信息泄漏。
测试标签与训练标签同口径（H3 投影 4m 框），无标注域差。
指标：mAP50/75/50-95、CLE、R25、按高度检出率、虚警、首次识别高度。
"""
import argparse, csv, os, sys, math, gc
from pathlib import Path
import numpy as np
import cv2

EDGES = [20, 30, 45, 60, 85, 120]
RUNS = os.path.join(REPO_ROOT, "yolo_runs")
SYN = os.path.join(REPO_ROOT, "yolo_syn")
MODELS = [
    ("base", "标准3DGS", os.path.join(RUNS, "syn_base", "weights", "best.pt")),
    ("ours", "本文方法(w=3)", os.path.join(RUNS, "syn_ours", "weights", "best.pt")),
    ("w5", "权重w=5", os.path.join(RUNS, "syn_w5", "weights", "best.pt")),
    ("focal", "Focal loss", os.path.join(RUNS, "syn_focal", "weights", "best.pt")),
    ("randroi", "随机ROI加权", os.path.join(RUNS, "syn_randroi", "weights", "best.pt")),
    ("gradscale", "梯度缩放", os.path.join(RUNS, "syn_gradscale", "weights", "best.pt")),
    ("budget", "预算x1.2", os.path.join(RUNS, "syn_budget", "weights", "best.pt")),
]
IMG_BASE = os.path.join(SYN, "images")
LBL_BASE = os.path.join(SYN, "labels")
VIEWS_CSV = os.path.join(SYN, "views_test_base.csv")
OUT = os.path.join(SYN, "eval")
CONF = 0.15


def iou(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0: return 0.0
    inter = (x1-x0)*(y1-y0)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/max(ua, 1e-9)


def read_gt(lbl_dir, name):
    p = os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt")
    if not os.path.exists(p): return None
    with open(p) as f:
        line = f.readline().split()
    return [float(x) for x in line[1:5]]  # cx,cy,w,h 归一化


def layer_of(alt):
    for i in range(len(EDGES)-1):
        hi = EDGES[i+1]
        is_last = (i == len(EDGES)-2)
        if EDGES[i] <= alt and (alt < hi or (is_last and alt <= hi)):
            return i
    return None


def main():
    from ultralytics import YOLO
    os.makedirs(OUT, exist_ok=True)
    # 虚拟高度：views_test.csv idx->alt（所有配置共享同一组 held-out 机位）
    idx2alt = {}
    with open(VIEWS_CSV, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            idx2alt[int(r["idx"])] = float(r["alt"])

    results = {}
    for key, label, w in MODELS:
        if not os.path.exists(w):
            print(f"!! 缺少模型 {w}，跳过"); continue
        img_dir = os.path.join(IMG_BASE, f"test_{key}")
        lbl_dir = os.path.join(LBL_BASE, f"test_{key}")
        if not os.path.isdir(img_dir):
            print(f"!! 缺少测试图 {img_dir}，跳过（先渲染 test_{key}）"); continue
        names = sorted(os.listdir(img_dir))
        imgs = []
        for nm in names:
            iidx = int(os.path.splitext(nm)[0].split("_")[1])
            gt = read_gt(lbl_dir, nm)
            if iidx in idx2alt and gt is not None:
                imgs.append((nm, idx2alt[iidx], gt))
        print(f"[{label}] 合成测试集 {len(imgs)} 张（held-out 机位）")
        data_yaml = os.path.join(SYN, f"data_test_{key}.yaml")
        if not os.path.exists(data_yaml):
            with open(data_yaml, "w") as f:
                f.write(f"path: {SYN}\ntrain: images/test_{key}\nval: images/test_{key}\n"
                        f"names:\n  0: helipad_H\n")
        m = YOLO(w)
        v = m.val(data=data_yaml, split="val", imgsz=1280, device=0, workers=0,
                  conf=0.001, iou=0.6, project=RUNS, name=f"eval_syn_{key}",
                  exist_ok=True, verbose=False, plots=False)
        map50, map75, map5095 = float(v.box.map50), float(v.box.map75), float(v.box.map)
        rows = []
        for nm, alt, gt in imgs:
            r = m.predict(source=os.path.join(img_dir, nm), imgsz=1280, conf=CONF,
                          iou=0.5, device=0, verbose=False)[0]
            H, W = r.orig_shape
            gx, gy, gw, gh = gt
            gb = [gx-gw/2, gy-gh/2, gx+gw/2, gy+gh/2]
            best = None; fp = 0; hit25 = False
            for b in r.boxes:
                x0, y0, x1, y1 = b.xyxy[0].tolist(); c = float(b.conf)
                pb = [x0/W, y0/H, x1/W, y1/H]
                pcx, pcy = 0.5*(pb[0]+pb[2]), 0.5*(pb[1]+pb[3])
                iov = iou(pb, gb)
                if math.hypot(pcx-gx, pcy-gy) < max(gw, 0.05) and iov > 0.1:
                    if best is None or c > best[0]:
                        best = (c, math.hypot(pcx-gx, pcy-gy), iov)
                else:
                    if c > 0.25: fp += 1
            det = best is not None
            hit25 = det and best[2] >= 0.25
            rows.append({"name": nm, "alt": round(alt,1), "det": int(det),
                         "hit25": int(hit25),
                         "conf": round(best[0],3) if det else 0.0,
                         "cerr": round(best[1],4) if det else "",
                         "iou": round(best[2],3) if det else "",
                         "fp": fp})
        results[key] = {"label": label, "rows": rows,
                        "map50": map50, "map75": map75, "map5095": map5095}
        del m
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        with open(os.path.join(OUT, f"per_view_{key}.csv"), "w", newline="",
                  encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, fieldnames=["name","alt","det","hit25","conf","cerr","iou","fp"])
            wr.writeheader(); wr.writerows(rows)
    if not results:
        print("没有可评测的模型，退出"); return

    print("\n========== 合成闭环（held-out 测试机位）结果 ==========")
    summary = []
    for key, d in results.items():
        rows = d["rows"]
        layer_rate = []
        for i in range(1, 5):  # 30-45 / 45-60 / 60-85 / 85-120（排除 20-30 超低空）
            sel = [r for r in rows if layer_of(r["alt"]) == i]
            if sel:
                rate = np.mean([r["det"] for r in sel])
                r25 = np.mean([r["hit25"] for r in sel])
                conf = np.mean([r["conf"] for r in sel if r["det"]]) if any(r["det"] for r in sel) else 0
                fp = np.mean([r["fp"] for r in sel])
                cle = np.mean([r["cerr"] for r in sel if r["det"]]) if any(r["det"] for r in sel) else None
                layer_rate.append((i, 0.5*(EDGES[i]+EDGES[i+1]), rate, r25, conf, len(sel), fp, cle))
        max_det = max([r["alt"] for r in rows if r["alt"] >= 30 and r["det"]], default=0)
        h50 = max([am for _, am, rate, *_ in layer_rate if rate >= 0.5], default=0)
        summary.append((key, d["label"], d["map50"], d["map75"], d["map5095"], max_det, h50, layer_rate))
        print(f"\n[{d['label']}] mAP50={d['map50']:.3f} mAP75={d['map75']:.3f} mAP50-95={d['map5095']:.3f}")
        print(f"  单张最高检出高度={max_det:.0f}m  检测率>=50%的最高高度={h50:.0f}m")
        for _, am, rate, r25, conf, n, fp, cle in layer_rate:
            s = f"  ~{am:.0f}m n={n:2d} 检出率 {rate*100:5.1f}%  R25 {r25*100:5.1f}%  置信度 {conf:.2f}  虚警 {fp:.2f}"
            if cle is not None: s += f"  CLE {cle*1000:.1f}(归一化x1000)"
            print(s)

    with open(os.path.join(OUT, "summary_synth.csv"), "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["model","mAP50","mAP75","mAP50-95","max_detect_alt_m","alt50_detect_m"])
        for key, lab, m50, m75, m5095, maxd, h50, _ in summary:
            wr.writerow([lab, round(m50,3), round(m75,3), round(m5095,3), round(maxd,1), round(h50,1)])

    # ---- 画图（学术图注，无口语词） ----
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
        if os.path.exists(fp):
            font_manager.fontManager.addfont(fp)
            matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name(); break
    matplotlib.rcParams["axes.unicode_minus"] = False
    colors = {"base": "#5b8db8", "ours": "#c0392b", "w5": "#2e8b57",
              "focal": "#8e8e8e", "randroi": "#b8860b", "gradscale": "#6a5acd",
              "budget": "#20b2aa"}

    fig, axs = plt.subplots(1, 3, figsize=(19, 5.2))
    for key, d in results.items():
        lr = next(s[7] for s in summary if s[0] == key)
        am = [x[1] for x in lr]; rt = [x[2]*100 for x in lr]
        cf = [x[4] for x in lr]; r25 = [x[3]*100 for x in lr]
        axs[0].plot(am, rt, "o-", color=colors[key], lw=2, ms=7, label=d["label"])
        axs[1].plot(am, r25, "s-", color=colors[key], lw=1.5, ms=5, label=d["label"])
        axs[2].plot(am, cf, "^-", color=colors[key], lw=1.5, ms=5, label=d["label"])
    axs[0].set_xlabel("虚拟高度 / m"); axs[0].set_ylabel("H 检出率 / %")
    axs[0].set_title("各高度档 H 检出率（合成测试机位）")
    axs[1].set_xlabel("虚拟高度 / m"); axs[1].set_ylabel("宽松召回 R25 / %")
    axs[1].set_title("Recall@IoU=0.25–高度")
    axs[2].set_xlabel("虚拟高度 / m"); axs[2].set_ylabel("平均置信度")
    axs[2].set_title("检出置信度–高度")
    for ax in axs:
        ax.grid(alpha=.3); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "synth_closed_detection.png"), dpi=140)
    print("\nSAVED synth_closed_detection.png")
    print("EVAL_SYNTH_CLOSED_DONE")


if __name__ == "__main__":
    main()
