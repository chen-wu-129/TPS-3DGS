# -*- coding: utf-8 -*-
"""paths.py - 仓库内所有路径的唯一出口

优先级：环境变量 > 仓库默认
  TPS_DATA_ROOT     数据集根目录   （默认 <repo>/data）
  TPS_OUT_ROOT      训练/评测输出  （默认 <repo>/outputs）
  TPS_RESULTS_ROOT  论文指标表     （默认 <repo>/results）
"""

import os
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
DATA_ROOT = os.environ.get("TPS_DATA_ROOT", os.path.join(REPO_ROOT, "data"))
OUT_ROOT = os.environ.get("TPS_OUT_ROOT", os.path.join(REPO_ROOT, "outputs"))
RESULTS_ROOT = os.environ.get("TPS_RESULTS_ROOT", os.path.join(REPO_ROOT, "results"))

for _p in (DATA_ROOT, OUT_ROOT, RESULTS_ROOT):
    os.makedirs(_p, exist_ok=True)
