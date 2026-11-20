# --- open-source path shim (added for release) ---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import REPO_ROOT, DATA_ROOT, OUT_ROOT, RESULTS_ROOT  # noqa: E402
# -------------------------------------------------
"""eval_radius_scan_v4.py —— 严格标志区半径扫描（用已有 PLY，只渲染不重训）。

动机：默认 H-PSNR 用 4m 半径投影方形（8m×8m），H 标志（约 4m 边长）仅占约 25%，
可能被周边地面的非特异全局增益污染。本脚本对每个留出视角只渲染一次，
在多个物理半径（1.5/2/3/4/6 m）的投影方形上分别计算 H-PSNR，
检验"增益是否随评测区收紧到真实标志而增大/消失"，作为位置特异性的独立证据。

输出：results\\v4_radius_scan_holdout.csv
"""
import os
import argparse
import numpy as np
import torch

from mini_train import load_scene, make_window, psnr
from eval_altitude import load_ply_splats, render_view
from read_altitude import read_xmp

ROOT = os.path.join(OUT_ROOT, "v4_fieldH")
DATA = os.path.join(DATA_ROOT, "fieldH")
OUT = RESULTS_ROOT
CONFIGS = ["base", "lossonly", "salonly", "ours", "w5",
           "focal", "randroi", "gradscale", "budget"]
RADII = [1.5, 2.0, 3.0, 4.0, 6.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--max_images", type=int, default=331)
    ap.add_argument("--sh_degree", type=int, default=2)
    ap.add_argument("--configs", nargs="*", default=CONFIGS)
    args = ap.parse_args()
    device = "cuda"

    print("加载场景...")
    data = load_scene(DATA, args.factor, args.max_images)
    n = len(data["pixels"])
    val_ids = list(range(7, n, 8))
    c2w_all = data["camtoworlds"]
    cam = c2w_all[:, :3, 3].numpy()
    pts = data["points"].numpy()

    alt = []
    for nm in data["names"]:
        rel, _, _ = read_xmp(os.path.join(DATA, "images", nm))
        alt.append(rel if rel is not None else np.nan)
    alt = np.array(alt)
    m = ~np.isnan(alt)
    scale = abs(np.polyfit(cam[m, 2], alt[m], 1)[0])
    from h3_util import correct_h3
    H3 = correct_h3(cam, pts, hax=(0, 1), b_axis=2)
    print(f"scale={scale:.2f}m/unit, val={len(val_ids)}")

    window = make_window(device)

    def scan_ply(ply):
        s = load_ply_splats(ply, args.sh_degree, device)
        # 每半径累计有效视角的 H-PSNR 列表
        acc = {r: [] for r in RADII}
        with torch.no_grad():
            for idx in val_ids:
                gt = data["pixels"][idx].to(device)
                w, h = data["wh"][idx]
                c2w = c2w_all[idx:idx + 1].to(device)
                K = data["Ks"][idx:idx + 1].to(device)
                pred, _ = render_view(s, c2w, K, w, h, args.sh_degree)
                viewmat = torch.linalg.inv(c2w)[0].cpu().numpy()
                pc = viewmat @ np.array([H3[0], H3[1], H3[2], 1.0])
                if pc[2] <= 0:
                    continue
                Kn = K[0].cpu().numpy()
                q = Kn @ pc[:3]
                u, v = q[0] / q[2], q[1] / q[2]
                if not (0 <= u <= w and 0 <= v <= h):
                    continue
                for r in RADII:
                    rpx = Kn[0, 0] * (r / scale) / pc[2]
                    x0, x1 = int(max(0, u - rpx)), int(min(w, u + rpx))
                    y0, y1 = int(max(0, v - rpx)), int(min(h, v + rpx))
                    if x1 - x0 > 3 and y1 - y0 > 3:
                        acc[r].append(psnr(pred[y0:y1, x0:x1],
                                           gt[y0:y1, x0:x1]).item())
        del s
        import gc; gc.collect(); torch.cuda.empty_cache()
        return {r: (float(np.mean(v)) if v else np.nan, len(v)) for r, v in acc.items()}

    rows = []
    results = {}
    for cfg in args.configs:
        ply = os.path.join(ROOT, f"seed0_{cfg}", "mini_trained.ply")
        if not os.path.exists(ply):
            print(f"[skip] {cfg} 无 PLY")
            continue
        print(f"评测 {cfg} ...")
        res = scan_ply(ply)
        results[cfg] = res
        row = {"config": cfg}
        for r in RADII:
            row[f"HPSNR_r{r}"] = round(res[r][0], 3)
            row[f"n_r{r}"] = res[r][1]
        rows.append(row)

    # 输出绝对表
    path = os.path.join(OUT, "v4_radius_scan_holdout.csv")
    import csv
    fields = ["config"] + [f"HPSNR_r{r}" for r in RADII] + [f"n_r{r}" for r in RADII]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for row in rows:
            wr.writerow(row)

    # 打印相对 base 的增益表（核心）
    print("\n===== H-PSNR 相对 base 的增益（dB）随评测半径变化 =====")
    header = "config   " + "".join([f"  r={r:<4}" for r in RADII])
    print(header)
    base = results.get("base", {})
    for cfg in args.configs:
        if cfg not in results:
            continue
        line = f"{cfg:<9}"
        for r in RADII:
            d = results[cfg][r][0] - base[r][0]
            line += f"  {d:+5.2f} "
        print(line)
    print(f"\nSAVED {path}")


if __name__ == "__main__":
    main()
