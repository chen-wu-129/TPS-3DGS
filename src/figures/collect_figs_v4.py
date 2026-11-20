# --- open-source path shim (added for release) ---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import REPO_ROOT, DATA_ROOT, OUT_ROOT, RESULTS_ROOT  # noqa: E402
# -------------------------------------------------
# -*- coding: utf-8 -*-
"""collect_figs_v4.py —— H3 返工后把重算的图统一归集到论文 figures 目录（图3/4/5/6）。
由 run_v4_figs_final.bat 在所有 GPU 流水线结束后调用；纯复制，不占 GPU。
"""
import os
import shutil

FIGD = os.path.join(OUT_ROOT, "figures")
DATA = RESULTS_ROOT
BFIG = os.path.join(REPO_ROOT, "budget_fig")

MAP = [
    (os.path.join(BFIG, "roi_adaptive.png"),
     os.path.join(FIGD, "fig3_roi_adaptive.png")),
    (os.path.join(DATA, "v4_fig5_zoom_H.png"),
     os.path.join(FIGD, "fig4_render_zoom.png")),
    (os.path.join(DATA, "v4_fig5_full.png"),
     os.path.join(FIGD, "fig4_render_full.png")),
    (os.path.join(BFIG, "v4_radial_density.png"),
     os.path.join(FIGD, "fig5_radial_density.png")),
    (os.path.join(BFIG, "v4_dose_curve_holdout.png"),
     os.path.join(FIGD, "fig6_dose_curve.png")),
]


def main():
    os.makedirs(FIGD, exist_ok=True)
    for src, dst in MAP:
        if os.path.exists(src):
            shutil.copyfile(src, dst)
            print(f"COPIED {os.path.basename(src)} -> {os.path.basename(dst)}")
        else:
            print(f"!! MISSING {src}")
    print("COLLECT_FIGS_DONE")


if __name__ == "__main__":
    main()
