# --- open-source path shim (added for release) ---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import REPO_ROOT, DATA_ROOT, OUT_ROOT, RESULTS_ROOT  # noqa: E402
# -------------------------------------------------
# -*- coding: utf-8 -*-
"""render_synthetic_v2.py —— sim→real 合成训练集（修正版）

相对 render_synthetic.py 的唯一本质改动：
    原版：target = H3 + 抖动(±4m)  → H 几乎永远落在画面正中（实测 cx/cy≈0.50，
          距中心中位数 0.017、95% 分位 0.045）；
          而真实照片 H 距中心中位数 0.133、95% 分位 0.429，cy 均值 0.344。
          → 检测器学到"目标在中心"的捷径，真实照片一旦目标偏离就失效。

    新版：先按真实分布采样 H 的目标成像位置 (u,v)，再迭代求解相机朝向，
          使 H3 恰好投影到该位置；并叠加小角度 roll。位姿生成与 PLY 无关，
          因此 base / ours / w5 三套机位完全一致（与原版同约定）。
"""
import argparse, csv, os, sys
from pathlib import Path
import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from mini_train import load_scene
from eval_altitude import load_ply_splats, render_view
from read_altitude import read_xmp

H3 = np.array([0.06973587, 0.04483227, 0.00587415])
B = 2
HALF_M = 2.0
NW, NH = 2048, 1152
N_ALT = 20
N_AZ = 16
R_KNOT_ALT = np.array([25.0, 37.5, 52.5, 72.5, 102.5])
R_KNOT_R = np.array([80.0, 71.0, 82.0, 101.0, 118.0])
SKY = np.array([0.80, 0.84, 0.88], dtype=np.float32)


def look_at_c2w(eye, target, up):
    z = target - eye; z /= np.linalg.norm(z)
    x = np.cross(z, up); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c = np.eye(4)
    c[:3, 0] = x; c[:3, 1] = y; c[:3, 2] = z; c[:3, 3] = eye
    return c


def project_pt(p, c2w, K):
    pc = np.linalg.inv(c2w) @ np.array([p[0], p[1], p[2], 1.0])
    if pc[2] <= 0.05:
        return None
    q = K @ pc[:3]
    return q[0] / q[2], q[1] / q[2]


def c2w_for_uv(eye, u_t, v_t, K, up, iters=12):
    """迭代调整视线目标点，使 H3 投影落在 (u_t, v_t)。"""
    depth = float(np.linalg.norm(H3 - eye))
    w_off = np.zeros(3)
    c2w = look_at_c2w(eye, H3, up)
    for _ in range(iters):
        uv = project_pt(H3, c2w, K)
        if uv is None:
            return None
        du, dv = u_t - uv[0], v_t - uv[1]
        if abs(du) < 1.0 and abs(dv) < 1.0:
            break
        ax = c2w[:3, 0]; ay = c2w[:3, 1]
        w_off += (-du / K[0, 0]) * depth * ax + (-dv / K[1, 1]) * depth * ay
        c2w = look_at_c2w(eye, H3 + w_off, up)
    uv = project_pt(H3, c2w, K)
    if uv is None or abs(uv[0] - u_t) > 3 or abs(uv[1] - v_t) > 3:
        return None
    return c2w


def gen_views(scale, intercept, alt_real, r_real, seed=0, n_alt=N_ALT, n_az=N_AZ):
    """返回 (eye, u_t, v_t, alt, r, az, idx, roll)。目标成像位置按真实分布采样。"""
    rng = np.random.RandomState(seed)
    alts = np.linspace(22.0, 120.0, n_alt)
    views = []
    idx = 0
    for li, alt in enumerate(alts):
        band = np.abs(alt_real - alt) <= 5.0
        if band.sum() >= 4:
            r_lo = float(np.percentile(r_real[band], 15))
            r_hi = float(np.percentile(r_real[band], 85))
        else:
            r_med = float(np.interp(alt, R_KNOT_ALT, R_KNOT_R))
            r_lo, r_hi = 0.7 * r_med, 1.2 * r_med
        if alt < 45.0:
            r_approach = 10.0 + 0.9 * max(alt - 22.0, 0.0)
            r_lo = min(r_lo, r_approach)
        r_lo, r_hi = max(r_lo, 10.0), min(r_hi, 150.0)
        off = (np.pi / n_az) if li % 2 else 0.0
        for k in range(n_az):
            frac = k / (n_az - 1)
            rbase = r_lo + (r_hi - r_lo) * frac if li % 2 == 0 \
                else r_hi - (r_hi - r_lo) * frac
            rj = rbase * (1.0 + 0.05 * (rng.rand() - 0.5) * 2)
            az = 2 * np.pi * k / n_az + off
            axis2 = (alt - intercept) / (-scale)
            eye = np.array([
                H3[0] + (rj / scale) * np.cos(az),
                H3[1] + (rj / scale) * np.sin(az),
                axis2,
            ])
            # 目标成像位置：贴近真实照片分布（cx≈0.49, cy≈0.34），并保留一定散布
            u_t = float(np.clip(rng.normal(0.49, 0.14), 0.10, 0.90)) * NW
            v_t = float(np.clip(rng.normal(0.34, 0.14), 0.06, 0.82)) * NH
            roll = float(np.deg2rad(rng.uniform(-8.0, 8.0)))
            views.append((eye, u_t, v_t, float(alt), rj, float(np.degrees(az)), idx, roll))
            idx += 1
    return views


def project(pts, c2w, K):
    vm = np.linalg.inv(c2w)
    us, vs, ds = [], [], []
    for p in pts:
        pc = vm @ np.array([p[0], p[1], p[2], 1.0])
        if pc[2] <= 0.05:
            return None
        q = K @ pc[:3]
        us.append(q[0] / q[2]); vs.append(q[1] / q[2]); ds.append(q[2])
    return us, vs, ds


def roll_c2w(c2w, ang):
    ca, sa = np.cos(ang), np.sin(ang)
    R = np.eye(4)
    R[0, 0], R[0, 1] = ca, -sa
    R[1, 0], R[1, 1] = sa, ca
    out = c2w.copy()
    out[:3, :3] = c2w[:3, :3] @ R[:3, :3]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(DATA_ROOT, "fieldH"))
    ap.add_argument("--ply", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_alt", type=int, default=N_ALT)
    ap.add_argument("--n_az", type=int, default=N_AZ)
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--max_images", type=int, default=331)
    ap.add_argument("--sh_degree", type=int, default=2)
    ap.add_argument("--root", default="yolo_syn_v2")
    ap.add_argument("--out_w", type=int, default=2048)
    ap.add_argument("--out_h", type=int, default=1152)
    args = ap.parse_args()
    device = "cuda"
    img_dir = os.path.join(args.root, "images", args.tag)
    lbl_dir = os.path.join(args.root, "labels", args.tag)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    data = load_scene(args.data_dir, args.factor, args.max_images)
    c2w_all = data["camtoworlds"].numpy()
    cam = c2w_all[:, :3, 3]
    alt = []
    for nm in data["names"]:
        rel, _, _ = read_xmp(os.path.join(args.data_dir, "images", nm))
        alt.append(rel if rel is not None else np.nan)
    alt = np.array(alt)
    m = ~np.isnan(alt)
    A = np.polyfit(cam[m, B], alt[m], 1)
    scale = abs(A[0]); intercept = A[1]
    print(f"scale={scale:.2f} intercept={intercept:.2f}")

    global NW, NH
    NW, NH = args.out_w, args.out_h
    _native_w = float(data["wh"][0][0] if not isinstance(data["wh"][0], int) else data["wh"][0])
    K = data["Ks"][0].numpy() * (NW / _native_w)
    K[2, 2] = 1.0

    fwd = c2w_all[:, :3, 2]
    to = H3 - cam
    valid = (fwd * to).sum(axis=1) > 0
    valid &= ~np.isnan(alt)
    r_all = np.hypot(cam[:, 0] - H3[0], cam[:, 1] - H3[1]) * scale
    alt_real, r_real = alt[valid], r_all[valid]

    views = gen_views(scale, intercept, alt_real, r_real,
                      seed=args.seed, n_alt=args.n_alt, n_az=args.n_az)
    print(f"虚拟机位 {len(views)} 个，加载 PLY: {args.ply}")
    s = load_ply_splats(args.ply, args.sh_degree, device)

    d = HALF_M / scale
    corners = [(H3[0] + sx * d, H3[1] + sy * d, H3[2]) for sx in (-1, 1) for sy in (-1, 1)]
    world_up = np.zeros(3); world_up[B] = -1.0
    rows, preview, kept, skipped = [], [], 0, 0
    with torch.no_grad():
        for vi, (eye, u_t, v_t, altm, r, azdeg, idx, roll) in enumerate(views):
            c2w = c2w_for_uv(eye, u_t, v_t, K, world_up)
            if c2w is None:
                skipped += 1
                continue
            c2w = roll_c2w(c2w, roll)
            c2w_t = torch.from_numpy(c2w).float()[None].to(device)
            K_t = torch.from_numpy(K).float()[None].to(device)
            pred, alpha = render_view(s, c2w_t, K_t, NW, NH, args.sh_degree)
            a = alpha.clamp(0, 1).cpu().numpy()
            if a.ndim == 2:
                a = a[..., None]
            im = (pred.cpu().numpy() * a + SKY[None, None, :] * (1 - a))
            im = (np.clip(im, 0, 1) * 255).astype(np.uint8)

            pr = project(corners, c2w, K)
            lab = ""
            inframe = False
            x0 = y0 = x1 = y1 = 0
            if pr is not None:
                us, vs, ds = pr
                x0, x1 = min(us), max(us); y0, y1 = min(vs), max(vs)
                if x1 > 0 and y1 > 0 and x0 < NW and y0 < NH:
                    x0 = max(0, x0 - 2); x1 = min(NW - 1, x1 + 2)
                    y0 = max(0, y0 - 2); y1 = min(NH - 1, y1 + 2)
                    if x1 - x0 >= 4 and y1 - y0 >= 4:
                        xc = 0.5 * (x0 + x1) / NW; yc = 0.5 * (y0 + y1) / NH
                        bw = (x1 - x0) / NW; bh = (y1 - y0) / NH
                        lab = f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"
                        inframe = True
            stem = f"syn_{idx:04d}"
            cv2.imwrite(os.path.join(img_dir, stem + ".jpg"),
                        cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            if inframe:
                with open(os.path.join(lbl_dir, stem + ".txt"), "w") as f:
                    f.write(lab)
                kept += 1
            rows.append({"idx": idx, "alt": round(altm, 1), "r": round(r, 1),
                         "az": round(azdeg, 1), "has_H": int(inframe)})
            if inframe and vi % 64 == 0:
                preview.append((altm, im.copy(), x0, y0, x1, y1))
            if (vi + 1) % 80 == 0:
                print(f"  {vi+1}/{len(views)}  kept={kept} skipped={skipped}")

    with open(os.path.join(args.root, f"views_{args.tag}.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=["idx", "alt", "r", "az", "has_H"])
        wr.writeheader(); wr.writerows(rows)
    print(f"{args.tag}: 渲染 {len(views)}，H在画面 {kept}，跳过 {skipped}")

    if preview:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pick = [preview[int(i * (len(preview) - 1) / 5)] for i in range(6)]
        fig, axs = plt.subplots(2, 3, figsize=(18, 7))
        for ax, (altm, im, x0, y0, x1, y1) in zip(axs.ravel(), pick):
            imc = im.copy()
            cv2.rectangle(imc, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
            ax.imshow(imc); ax.set_title(f"{altm:.0f} m"); ax.axis("off")
        fig.suptitle(f"synthetic v2 {args.tag} (green = auto H label)")
        fig.tight_layout(); fig.savefig(os.path.join(args.root, f"preview_{args.tag}.png"), dpi=120)
    print("RENDER_SYN_V2_DONE", args.tag)


if __name__ == "__main__":
    main()