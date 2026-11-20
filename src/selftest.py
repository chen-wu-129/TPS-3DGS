# --- open-source path shim (added for release) ---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import REPO_ROOT, DATA_ROOT, OUT_ROOT, RESULTS_ROOT  # noqa: E402
# -------------------------------------------------
# -*- coding: utf-8 -*-
"""selftest.py - 不需要 GPU 的数据/几何自检

检查三件事：
  1) 场景能否加载（位姿、内参、图片、初始点云）；
  2) 高度表能否读出（XMP 缺失时走 altitude.csv）；
  3) 着陆标志 H 的三维定位与屏幕 ROI 投影在各高度层是否落在画面内、半径是否随高度收缩。
用法：python src/selftest.py --data_dir data/fieldH --factor 8 --max_images 60
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mini_train import load_scene  # noqa: E402
from read_altitude import read_xmp  # noqa: E402
from train_ours import estimate_h3d, roi_weight_map  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(DATA_ROOT, "fieldH"))
    ap.add_argument("--factor", type=int, default=8)
    ap.add_argument("--max_images", type=int, default=60)
    ap.add_argument("--radius_m", type=float, default=6.0)
    ap.add_argument("--weight", type=float, default=3.0)
    args = ap.parse_args()

    print("[1/3] 加载场景 %s" % args.data_dir)
    data = load_scene(args.data_dir, args.factor, args.max_images)
    print("      视角 %d，初始点 %d，图像 %dx%d"
          % (len(data["pixels"]), data["points"].shape[0],
             data["wh"][0][0], data["wh"][0][1]))

    print("[2/3] 读取高度")
    alt = [read_xmp(os.path.join(args.data_dir, "images", n))[0] for n in data["names"]]
    ok = [a for a in alt if a is not None]
    assert ok, "高度全缺：检查 altitude.csv 或在 data_dir 放带 XMP 的原片"
    print("      %d/%d 张有高度，范围 %.1f–%.1f m"
          % (len(ok), len(alt), min(ok), max(ok)))

    print("[3/3] H 定位 + ROI 投影")
    H3, vax, scale = estimate_h3d(data, args.data_dir)
    print("      1 归一化单位 = %.2f m" % scale)
    hax = [a for a in range(3) if a != vax]
    k0 = data["Ks"][0, 0, 0].item()
    r_norm = args.radius_m / scale
    inside, radii = 0, []
    for i, nm in enumerate(data["names"]):
        c2w = data["camtoworlds"][i:i + 1]
        K = data["Ks"][i:i + 1]
        w, h = data["wh"][i]
        viewmat = torch.linalg.inv(c2w)[0].numpy()
        pc = viewmat @ np.array([H3[0], H3[1], H3[2], 1.0])
        if pc[2] <= 0:
            continue
        q = K[0].numpy() @ pc[:3]
        u, v = q[0] / q[2], q[1] / q[2]
        rpx = K[0, 0, 0].item() * r_norm / pc[2]
        if 0 <= u < w and 0 <= v < h:
            inside += 1
            radii.append((alt[i] or 0.0, rpx))
    radii.sort()
    if radii:
        lo, hi = radii[0], radii[-1]
        print("      画面内 %d/%d 视角；ROI 屏幕半径 %.0f px(@%.0f m) → %.0f px(@%.0f m)"
              % (inside, len(data["names"]), lo[1], lo[0], hi[1], hi[0]))
    W = roi_weight_map(data["camtoworlds"][0:1], data["Ks"][0:1],
                       int(data["wh"][0][0]), int(data["wh"][0][1]),
                       H3, scale, args.radius_m, args.weight, "cpu")
    print("      权重图 %s，最大值 %.2f，均值 %.3f"
          % (tuple(W.shape), W.max().item(), W.mean().item()))
    print("SELFTEST_OK")


if __name__ == "__main__":
    main()
