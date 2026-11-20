# --- open-source path shim (added for release) ---
import os
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import REPO_ROOT, DATA_ROOT, OUT_ROOT, RESULTS_ROOT  # noqa: E402
# -------------------------------------------------
# -*- coding: utf-8 -*-
"""render_fps.py —— 实测三个最终模型的渲染帧率与推理显存峰值。
分辨率：512x288（训练/评测）与 2048x1152（合成数据，与原始航拍一致）。
复用 eval_altitude.load_ply_splats / render_view，保证渲染路径与评测完全一致。
"""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ref"))
from mini_train import load_scene  # noqa: E402
from eval_altitude import load_ply_splats, render_view  # noqa: E402

DATA = os.path.join(DATA_ROOT, "fieldH")
PLYS = [("baseline", os.path.join(OUT_ROOT, "fieldH/mini_trained.ply")),
        ("w3", os.path.join(OUT_ROOT, "fieldH_ours/mini_trained.ply")),
        ("w5", os.path.join(OUT_ROOT, "fieldH_ours_w5/mini_trained.ply"))]
SH = 2
DEVICE = "cuda"


def bench(s, c2ws, Ks, w, h, warmup, nframe):
    nv = len(c2ws)
    with torch.no_grad():
        for i in range(warmup):
            idx = i % nv
            render_view(s, c2ws[idx:idx + 1], Ks[idx:idx + 1], w, h, SH)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for i in range(nframe):
            idx = i % nv
            render_view(s, c2ws[idx:idx + 1], Ks[idx:idx + 1], w, h, SH)
        torch.cuda.synchronize()
        dt = time.time() - t0
        mem = torch.cuda.max_memory_allocated() / 1024 ** 3
    return nframe / dt, mem


def main():
    print("加载场景...", flush=True)
    data = load_scene(DATA, 4, 331)
    n = len(data["pixels"])
    val = list(range(7, n, 8))
    c2ws = data["camtoworlds"][val].to(DEVICE)
    Ks512 = data["Ks"][val].to(DEVICE)
    Ks2048 = Ks512.clone() * 4.0
    Ks2048[:, 2, 2] = 1.0
    print(f"测试视角 {len(val)} 个；GPU={torch.cuda.get_device_name(0)}", flush=True)

    print("=" * 70)
    print(f"{'模型':<10}{'高斯数':>11}{'FPS512':>10}{'显存512':>10}{'FPS2048':>10}{'显存2048':>10}")
    print("=" * 70)
    for name, p in PLYS:
        s = load_ply_splats(p, SH, DEVICE)
        ng = int(s["means"].shape[0])
        fps1, m1 = bench(s, c2ws, Ks512, 512, 288, 10, 120)
        fps2, m2 = bench(s, c2ws, Ks2048, 2048, 1152, 5, 40)
        print(f"{name:<10}{ng:>11,}{fps1:>9.1f}f{m1:>8.2f}G{fps2:>9.1f}f{m2:>8.2f}G", flush=True)
        del s
        torch.cuda.empty_cache()
    print("FPS_BENCH_DONE")


if __name__ == "__main__":
    main()
