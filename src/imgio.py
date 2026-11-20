# -*- coding: utf-8 -*-
"""imgio.py —— 兼容非 ASCII 路径的图像读取（Windows 上 cv2.imread 读不了中文路径）。"""

import cv2
import numpy as np


def imread_any(path, flags=cv2.IMREAD_COLOR):
    """先按普通路径读，失败则用 np.fromfile + imdecode 兜底；返回 None 表示读不到。"""
    img = cv2.imread(str(path), flags)
    if img is not None:
        return img
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except (IOError, OSError):
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)
