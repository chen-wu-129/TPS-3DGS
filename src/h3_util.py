# --- open-source path shim (added for release) ---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import REPO_ROOT, DATA_ROOT, OUT_ROOT, RESULTS_ROOT  # noqa: E402
# -------------------------------------------------
# -*- coding: utf-8 -*-
"""h3_util.py —— 权威的目标中心（H3）估计。

水平位置：相机光心水平坐标均值（环绕轨迹质心，已逐视角验证落在 H 正上方）。
竖直位置：H 水平邻域内 SfM 点云的地面高度（多半径直方图主峰的中位）。

严禁改用全局点云低分位数（np.percentile(pts,5)）：看台/树/噪声等高于地面的
点会把它拉偏到地面上方约 7m，导致屏幕 ROI / 标签框整体漂移到目标远侧。
该 bug 已于 2026-10-24 用真实航拍照片逐视角确认并修正。
"""
import numpy as np

# 本场景（fieldH）经全视角验证的权威常量；仅用于不加载场景数据的离线脚本。
SCALE_M = 99.433
H3_FIXED = np.array([0.06973587, 0.04483227, 0.00587415])


def infer_vax(camtoworlds, default=2, min_abs=0.5):
    """从归一化相机姿态推断竖直轴（相机 y 向下，世界 up = -R[:,1]）。

    公开数据没有 XMP，不能靠 read_xmp 拟合高度；归一化后相机 up 的主方向
    仍然可用，用于给 SalienceStrategy / ROI 选择两个水平轴。
    """
    R = np.asarray(camtoworlds, dtype=np.float64)[:, :3, :3]
    up = np.mean(-R[:, :, 1], axis=0)
    if not np.all(np.isfinite(up)) or np.linalg.norm(up) < 1e-9:
        return int(default)
    ax = int(np.argmax(np.abs(up)))
    return ax if abs(up[ax]) >= float(min_abs) else int(default)


def metric_scale_from_T1(T1):
    """归一化变换 T1 的缩放列 -> 1 归一化单位对应的米数。"""
    T1 = np.asarray(T1, dtype=np.float64)
    return 1.0 / max(float(np.linalg.norm(T1[:3, 0])), 1e-12)


def correct_h3(cam, pts, hax=(0, 1), b_axis=2, scale=SCALE_M):
    """相机位姿 cam(N,3)、SfM 点云 pts(M,3) -> 修正后的 H3(3,)。

    hax：两个水平轴；b_axis：竖直轴。本场景 b_axis=2、hax=(0,1)。
    """
    cam = np.asarray(cam, dtype=np.float64)
    pts = np.asarray(pts, dtype=np.float64)
    hx = float(cam[:, hax[0]].mean())
    hy = float(cam[:, hax[1]].mean())
    d0 = pts[:, hax[0]] - hx
    d1 = pts[:, hax[1]] - hy
    dist = np.sqrt(d0 * d0 + d1 * d1)
    zgs = []
    for R in (4.0, 6.0, 8.0, 10.0):
        zv = pts[dist <= R / scale, b_axis]
        if len(zv) < 20:
            continue
        hist, edges = np.histogram(zv, bins=40)
        pk = edges[int(np.argmax(hist))]
        band = zv[(zv >= pk - 0.005) & (zv <= pk + 0.005)]
        zgs.append(float(np.median(band)) if len(band) else float(np.median(zv)))
    if not zgs:
        z_g = float(np.median(pts[:, b_axis]))
    else:
        z_g = float(np.median(zgs))
    H3 = np.zeros(3, dtype=np.float64)
    H3[hax[0]] = hx
    H3[hax[1]] = hy
    H3[b_axis] = z_g
    return H3
