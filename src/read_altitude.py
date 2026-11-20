# --- open-source path shim (added for release) ---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import REPO_ROOT, DATA_ROOT, OUT_ROOT, RESULTS_ROOT  # noqa: E402
# -------------------------------------------------
# -*- coding: utf-8 -*-
"""read_altitude.py - 飞行高度读取（相对起飞点，米）

两种来源，按优先级：
1) DJI 原片的 XMP（文件头部 RelativeAltitude 字段）；
2) 仓库发布的副本照片已去除 EXIF，此时回落到 data/fieldH/altitude.csv。
"""
import csv
import glob
import os
import re

PAT_REL = re.compile(rb"RelativeAltitude[^0-9+-]*([+-]?\d+\.?\d*)")
PAT_ABS = re.compile(rb"AbsoluteAltitude[^0-9+-]*([+-]?\d+\.?\d*)")

_ALT_CSV = None


def _read_xmp_raw(path):
    with open(path, "rb") as f:
        raw = f.read(200000)  # XMP 在文件头部
    rel = PAT_REL.search(raw)
    ab = PAT_ABS.search(raw)
    return (float(rel.group(1)) if rel else None,
            float(ab.group(1)) if ab else None,
            b"RelativeAltitude" in raw)


def _read_csv(path):
    tab = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("rel_alt_m"):
                tab[os.path.basename(row["image"])] = float(row["rel_alt_m"])
    return tab


def _nearest_alt_csv(path):
    """场景自带的 altitude.csv：先看图片同级，再看上一级目录"""
    d = os.path.dirname(os.path.abspath(path))
    for c in (os.path.join(d, "altitude.csv"), os.path.join(os.path.dirname(d), "altitude.csv")):
        if os.path.exists(c):
            return c
    return None


def _load_alt_csv():
    global _ALT_CSV
    if _ALT_CSV is not None:
        return _ALT_CSV
    _ALT_CSV = {}
    cands = [os.environ.get("TPS_ALT_CSV"),
             os.path.join(DATA_ROOT, "fieldH", "altitude.csv"),
             os.path.join(DATA_ROOT, "altitude.csv")]
    for c in cands:
        if c and os.path.exists(c):
            _ALT_CSV.update(_read_csv(c))
            break
    return _ALT_CSV


def read_xmp(path):
    """返回 (相对高度, 绝对高度, 是否读到 XMP)"""
    try:
        rel, ab, has = _read_xmp_raw(path)
    except Exception:
        rel, ab, has = None, None, False
    if rel is None:
        c = _nearest_alt_csv(path)
        rel = _read_csv(c).get(os.path.basename(path)) if c else None
        if rel is None:
            rel = _load_alt_csv().get(os.path.basename(path))
        if rel is not None:
            has = True
    return rel, ab, has


def scan(folder, pattern="*.JPG", n=10):
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    print("\n=== %s（共 %d 张，抽样 %d）===" % (folder, len(files), min(n, len(files))))
    for fp in files[:n]:
        rel, ab, has = read_xmp(fp)
        print("  %-42s 相对高度=%s 海拔=%s 含XMP=%s" % (os.path.basename(fp)[:40], rel, ab, has))
    rels = [read_xmp(fp)[0] for fp in files]
    rels = [r for r in rels if r is not None]
    if rels:
        print("  >> 全量 %d 张相对高度(m): min=%.1f 中位=%.1f max=%.1f"
              % (len(rels), min(rels), sorted(rels)[len(rels) // 2], max(rels)))
    else:
        print("  >> 未读到高度（EXIF 已丢失且 altitude.csv 不匹配）")
    return files


if __name__ == "__main__":
    scan(os.path.join(DATA_ROOT, "fieldH", "images"), "*.JPG", n=5)
