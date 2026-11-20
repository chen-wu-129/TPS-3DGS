# --- open-source path shim (added for release) ---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import REPO_ROOT, DATA_ROOT, OUT_ROOT, RESULTS_ROOT  # noqa: E402
from imgio import imread_any  # noqa: E402
# -------------------------------------------------
# -*- coding: utf-8 -*-
"""
mini_train.py —— 3D 高斯泼溅（3DGS）最小可研究训练脚本
====================================================
这是给论文实验台准备的"教学版"训练器：单文件、每一步都可打印、可修改。
和荔枝（LichtFeld）GUI 的区别：
  - 荔枝：黑盒，点按钮，改不了方法；
  - 本脚本：训练循环全部在你眼前，后面的"高度分层、显著性预算"都在这里改。

输入：荔枝导出的工程目录（里面有 images/ 和 sparse/0/，COLMAP 文本格式）
输出：训练后的高斯点云 .ply（可拖进 SuperSplat 看）+ 训练/测试指标

用法（先用小子集快速验证）：
  python mini_train.py --data_dir "./data/fieldH/images" \
      --factor 8 --max_images 120 --max_steps 3000 --out results\fuyao_mini
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# 复用 gsplat 官方的场景归一化函数（仅做尺度归一化，物理意义明确）
sys.path.insert(0, str(Path(__file__).parent / "ref"))
from normalize import (  # noqa: E402
    align_principal_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)
from gsplat import rasterization, export_splats  # noqa: E402
from gsplat.strategy import DefaultStrategy  # noqa: E402


# ---------------------------------------------------------------------------
# 1. 解析 COLMAP 文本格式（cameras.txt / images.txt / points3D.txt）
# ---------------------------------------------------------------------------
def read_colmap_txt(sparse_dir):
    # ---- cameras.txt:  CAM_ID MODEL WIDTH HEIGHT fx fy cx cy ----
    cameras = {}
    with open(os.path.join(sparse_dir, "cameras.txt"), "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            assert p[1] == "PINHOLE", f"只支持 PINHOLE，当前是 {p[1]}"
            cameras[p[0]] = {
                "w": int(p[2]), "h": int(p[3]),
                "fx": float(p[4]), "fy": float(p[5]),
                "cx": float(p[6]), "cy": float(p[7]),
            }

    # ---- images.txt: 位姿行 10 列（ID QW QX QY QZ TX TY TZ CAM NAME），2D 点行跳过 ----
    # 兼容两种写法：COLMAP 原始“位姿行 + 点行配对”，以及只保留位姿（空行占位）的精简版
    images = []
    with open(os.path.join(sparse_dir, "images.txt"), "r", encoding="utf-8") as f:
        lines = [l for l in f if not l.startswith("#") and l.strip()]
    for ln in lines:
        p = ln.split()
        if len(p) != 10 or not p[0].isdigit():
            continue  # 2D 点行
        # q = (qw, qx, qy, qz), t = (tx, ty, tz)
        q = np.array([float(x) for x in p[1:5]])
        t = np.array([float(x) for x in p[5:8]])
        qw, qx, qy, qz = q
        R = np.array([
            [1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
            [2 * (qx*qy + qz*qw), 1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qx*qw)],
            [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx*qx + qy*qy)],
        ])
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        images.append({
            "w2c": w2c.astype(np.float32),
            "cam_id": p[8],
            "name": p[9],
        })

    # ---- points3D.txt:  POINT_ID X Y Z R G B ERROR ... ----
    pts, rgb = [], []
    with open(os.path.join(sparse_dir, "points3D.txt"), "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            pts.append([float(p[1]), float(p[2]), float(p[3])])
            rgb.append([float(p[4]), float(p[5]), float(p[6])])
    points = np.asarray(pts, dtype=np.float32)
    points_rgb = np.asarray(rgb, dtype=np.float32)
    return cameras, images, points, points_rgb


# ---------------------------------------------------------------------------
# 2. 加载场景：位姿、内参、图片（降采样）、初始点云；归一化到单位尺度
# ---------------------------------------------------------------------------
def load_scene(data_dir, factor, max_images):
    sparse_dir = os.path.join(data_dir, "sparse", "0")
    if not os.path.isdir(sparse_dir):
        sparse_dir = os.path.join(data_dir, "sparse")
    cameras, images, points, points_rgb = read_colmap_txt(sparse_dir)

    images = sorted(images, key=lambda x: x["name"])
    if max_images and len(images) > max_images:
        # 均匀抽 max_images 张，保证视角覆盖整个场景
        idx = np.linspace(0, len(images) - 1, max_images).astype(int)
        images = [images[i] for i in idx]

    w2c_mats = np.stack([im["w2c"] for im in images], 0)
    camtoworlds = np.linalg.inv(w2c_mats)
    raw_cam_pos = camtoworlds[:, :3, 3].copy()  # 归一化前的相机位置（米），高度分层用

    # 每个视角的内参（荔枝工程只有一个相机）
    Ks, wh = [], []
    for im in images:
        c = cameras[im["cam_id"]]
        K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]],
                     dtype=np.float32)
        K[:2] /= factor  # 只缩放 fx/fy/cx/cy，K[2,2] 必须保持为 1
        Ks.append(K)
        wh.append((c["w"] // factor, c["h"] // factor))

    # 场景归一化（和 gsplat 官方一致）：缩放到单位尺度、对齐主轴
    T1 = similarity_from_cameras(camtoworlds)
    camtoworlds = transform_cameras(T1, camtoworlds)
    points = transform_points(T1, points)
    T2 = align_principal_axes(points)
    camtoworlds = transform_cameras(T2, camtoworlds)
    points = transform_points(T2, points)
    scene_scale = 1.0

    # 读图 + 降采样（小图直接放内存，速度快）
    img_dir = os.path.join(data_dir, "images")
    pixels = []
    for n, im in enumerate(images):
        im_path = os.path.join(img_dir, im["name"])
        im_bgr = imread_any(im_path)
        if im_bgr is None:
            raise FileNotFoundError(im_path)
        w, h = wh[n]
        im_bgr = cv2.resize(im_bgr, (w, h), interpolation=cv2.INTER_AREA)
        pixels.append(cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB) / 255.0)
        if (n + 1) % 20 == 0:
            print(f"  读图 {n+1}/{len(images)}")

    out = {
        "camtoworlds": torch.from_numpy(camtoworlds).float(),
        "Ks": torch.from_numpy(np.stack(Ks, 0)).float(),
        "wh": wh,
        "pixels": [torch.from_numpy(p).float() for p in pixels],
        "points": torch.from_numpy(points).float(),
        "points_rgb": torch.from_numpy(points_rgb / 255.0).float(),
        "raw_cam_pos": torch.from_numpy(raw_cam_pos).float(),
        "names": [im["name"] for im in images],
        "scene_scale": scene_scale,
        # 保留归一化变换，供公开数据把外部给定的目标/尺度映射到训练坐标
        "T1": T1,
        "T2": T2,
        "metric_scale": 1.0 / max(float(np.linalg.norm(T1[:3, 0])), 1e-12),
    }
    return out


# ---------------------------------------------------------------------------
# 3. 初始高斯：用 SfM 点云初始化（位置、颜色、按近邻距离定尺度）
# ---------------------------------------------------------------------------
def rgb_to_sh(rgb):
    C0 = 0.28209479177387814
    return torch.clamp((rgb - 0.5) / C0, min=-1.0, max=1.0)


def init_splats(points, points_rgb, sh_degree, device, max_init=50000):
    N = points.shape[0]
    # SfM 点可能上百万：先随机下采样到 max_init，再用 KDTree 查近邻（避免 N×N 爆显存）
    if N > max_init:
        rng = np.random.RandomState(42)
        sel = rng.choice(N, max_init, replace=False)
        points = points[sel]
        points_rgb = points_rgb[sel]
        N = max_init
    pts_np = points.numpy()
    from scipy.spatial import cKDTree
    tree = cKDTree(pts_np)
    d, _ = tree.query(pts_np, k=5)  # 自身 + 4 个最近邻
    nn_dist = d[:, 1:].mean(axis=1)
    scales = torch.log(torch.from_numpy(
        nn_dist[:, None].repeat(3, axis=1)).float().clamp_min(1e-6))
    colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))
    colors[:, 0] = rgb_to_sh(points_rgb)
    # 四元数必须归一化（否则各向异性协方差定义不成立）
    quats = torch.nn.functional.normalize(torch.rand(N, 4), dim=-1)
    splats = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(points),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((N,), 0.1))),
        "sh0": torch.nn.Parameter(colors[:, :1]),
        "shN": torch.nn.Parameter(colors[:, 1:]),
    }).to(device)
    return splats


# ---------------------------------------------------------------------------
# 4. 结构相似性 SSIM（论文指标之一，这里用 11x11 高斯窗实现，无需额外库）
# ---------------------------------------------------------------------------
def make_window(device):
    coords = torch.arange(11, dtype=torch.float32, device=device) - 5
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g /= g.sum()
    w = (g[:, None] * g[None, :])[None, None].repeat(3, 1, 1, 1)
    return w.contiguous()


def ssim(x, y, w):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu1 = F.conv2d(x, w, padding=5, groups=3)
    mu2 = F.conv2d(y, w, padding=5, groups=3)
    v1 = F.conv2d(x * x, w, padding=5, groups=3) - mu1 ** 2
    v2 = F.conv2d(y * y, w, padding=5, groups=3) - mu2 ** 2
    c12 = F.conv2d(x * y, w, padding=5, groups=3) - mu1 * mu2
    return ((2*mu1*mu2 + C1) * (2*c12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (v1 + v2 + C2))


def psnr(a, b):
    return -10 * torch.log10(F.mse_loss(a, b))


# ---------------------------------------------------------------------------
# 5. 训练主循环
# ---------------------------------------------------------------------------
def train(args):
    device = "cuda"
    print(f"[1/4] 加载场景: {args.data_dir}")
    data = load_scene(args.data_dir, args.factor, args.max_images)
    n = len(data["pixels"])
    if args.all_train:
        val_ids = set()
        train_ids = list(range(n))
        print(f"  共 {n} 张视角：全量训练（held-out 评测交由独立架次数据）")
    else:
        val_ids = set(range(7, n, 8))
        train_ids = [i for i in range(n) if i not in val_ids]
        print(f"  共 {n} 张视角：训练 {len(train_ids)}，测试 {len(val_ids)}")
    print(f"  初始 SfM 点数: {data['points'].shape[0]}")
    # 事件计数需要的 H3（与本文方法同一几何先验，用于 baseline 的 ROI 归属对照）
    H3, vax, scale = None, None, None
    if args.event_log:
        from train_ours import estimate_h3d
        H3, vax, scale = estimate_h3d(data, args.data_dir)
        hax = [a for a in range(3) if a != vax]
        _r_norm = args.h_radius_m / scale

    print("[2/4] 初始化高斯 + 致密化策略")
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"  随机种子 seed={args.seed}（控制初始四元数、视角采样、致密化随机）")
    splats = init_splats(data["points"], data["points_rgb"], args.sh_degree, device)
    opt = {
        k: torch.optim.Adam([{"params": splats[k], "lr": lr}], eps=1e-15)
        for k, lr in zip(
            ["means", "scales", "quats", "opacities", "sh0", "shN"],
            [1.6e-4, 5e-3, 1e-3, 5e-2, 2.5e-3, 2.5e-3 / 20],
        )
    }
    strategy = DefaultStrategy(verbose=False,
                               refine_stop_iter=args.refine_stop)
    strategy.check_sanity(splats, opt)
    state = strategy.initialize_state(scene_scale=1.0)
    window = make_window(device)
    means_sched = torch.optim.lr_scheduler.ExponentialLR(
        opt["means"], gamma=0.01 ** (1.0 / args.max_steps))

    os.makedirs(args.out, exist_ok=True)

    def save_ply(path, sp):
        export_splats(means=sp["means"], scales=sp["scales"],
                      quats=sp["quats"], opacities=sp["opacities"],
                      sh0=sp["sh0"], shN=sp["shN"],
                      format="ply", save_to=path)

    print(f"[3/4] 开始训练 {args.max_steps} 步")
    if args.event_log:
        import csv as _csv3
        _ef = open(os.path.join(args.out, "event_trace.csv"), "w", newline="")
        _ew = _csv3.writer(_ef)
        _ew.writerow(["step", "n_gauss", "n_roi3d", "delta_gauss",
                      "delta_roi3d", "clone_roi", "split_roi",
                      "opac_roi_mean", "opac_out_mean"])
    t0 = time.time()
    for step in range(args.max_steps):
        idx = train_ids[np.random.randint(len(train_ids))]
        gt = data["pixels"][idx].to(device)
        w, h = data["wh"][idx]
        c2w = data["camtoworlds"][idx:idx+1].to(device)
        K = data["Ks"][idx:idx+1].to(device)

        if args.event_log:
            n_eb = len(splats["means"])
            with torch.no_grad():
                d0 = splats["means"][:, hax[0]] - H3[hax[0]]
                d1 = splats["means"][:, hax[1]] - H3[hax[1]]
                _rm_b = torch.sqrt(d0 * d0 + d1 * d1) <= _r_norm
                n_roi_b = int(_rm_b.sum().item())

        colors = torch.cat([splats["sh0"], splats["shN"]], 1)
        renders, alphas, info = rasterization(
            means=splats["means"], quats=splats["quats"],
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=colors,
            viewmats=torch.linalg.inv(c2w), Ks=K,
            width=w, height=h,
            sh_degree=min(step // 1000, args.sh_degree),
            absgrad=True, packed=False,
        )
        pred = renders[0].clamp(0, 1)
        l1 = (1 - 0.2) * F.l1_loss(pred, gt)
        gt_t = gt.permute(2, 0, 1)[None]
        pr_t = pred.permute(2, 0, 1)[None]
        l_ssim = 0.2 * (1 - ssim(pr_t, gt_t, window).mean())
        loss = l1 + l_ssim

        strategy.step_pre_backward(params=splats, optimizers=opt,
                                   state=state, step=step, info=info)
        loss.backward()
        for o in opt.values():
            o.step()
            o.zero_grad(set_to_none=True)
        means_sched.step()
        strategy.step_post_backward(params=splats, optimizers=opt,
                                    state=state, step=step, info=info,
                                    packed=False)

        if args.event_log and step % 100 == 0:
            with torch.no_grad():
                d0 = splats["means"][:, hax[0]] - H3[hax[0]]
                d1 = splats["means"][:, hax[1]] - H3[hax[1]]
                _rm_a = torch.sqrt(d0 * d0 + d1 * d1) <= _r_norm
                n_roi_a = int(_rm_a.sum().item())
                opa = torch.sigmoid(splats["opacities"]).squeeze()
                opac_roi = float(opa[_rm_a].mean().item()) if n_roi_a else float("nan")
                opac_out = float(opa[~_rm_a].mean().item()) if (~_rm_a).any() else float("nan")
                _ew.writerow([step, len(splats["means"]), n_roi_a,
                              len(splats["means"]) - n_eb, n_roi_a - n_roi_b,
                              0, 0, f"{opac_roi:.4f}", f"{opac_out:.4f}"])
                _ef.flush()
            n_eb = len(splats["means"])

        if step % 250 == 0 or step == args.max_steps - 1:
            mem = torch.cuda.max_memory_allocated() / 1024 ** 3
            print(f"  step {step:5d} | loss {loss.item():.4f} | "
                  f"PSNR {psnr(pred, gt).item():.2f} | 高斯 {len(splats['means']):,} | "
                  f"显存 {mem:.1f}GB | {step/max(time.time()-t0,1):.1f} it/s")

        # 每 5000 步存档一次（防中断丢成果）
        if step > 0 and step % 5000 == 0:
            ck = os.path.join(args.out, f"ckpt_{step}.ply")
            save_ply(ck, splats)
            print(f"  [存档] {ck}（{len(splats['means']):,} 高斯）")

    # 可选：按不透明度裁剪到高斯上限（控制最终模型大小/渲染显存）
    if args.max_gaussians > 0 and len(splats["means"]) > args.max_gaussians:
        with torch.no_grad():
            opa = torch.sigmoid(splats["opacities"]).squeeze()
            keep = torch.topk(opa, args.max_gaussians).indices.sort().values
        for k in splats:
            splats[k] = torch.nn.Parameter(splats[k].data[keep])
        print(f"  按不透明度裁剪到 {args.max_gaussians:,} 高斯")

    # 测试集评测
    print("[4/4] 测试视角评测（论文的尺子）")
    psnrs, ssims = [], []
    if val_ids:
        with torch.no_grad():
            for idx in sorted(val_ids):
                gt = data["pixels"][idx].to(device)
                w, h = data["wh"][idx]
                c2w = data["camtoworlds"][idx:idx+1].to(device)
                K = data["Ks"][idx:idx+1].to(device)
                colors = torch.cat([splats["sh0"], splats["shN"]], 1)
                renders, _, _ = rasterization(
                    means=splats["means"], quats=splats["quats"],
                    scales=torch.exp(splats["scales"]),
                    opacities=torch.sigmoid(splats["opacities"]),
                    colors=colors, viewmats=torch.linalg.inv(c2w), Ks=K,
                    width=w, height=h, sh_degree=args.sh_degree)
                pred = renders[0].clamp(0, 1)
                psnrs.append(psnr(pred, gt).item())
                ssims.append(ssim(pred.permute(2, 0, 1)[None],
                                  gt.permute(2, 0, 1)[None], window).mean().item())
        print(f"  测试集 PSNR = {np.mean(psnrs):.2f} dB")
        print(f"  测试集 SSIM = {np.mean(ssims):.4f}")
    else:
        print("  全量训练模式：本模型不做同架次评测（held-out 评测由独立架次数据完成）")
    if args.event_log:
        _ef.close()

    os.makedirs(args.out, exist_ok=True)
    out_ply = os.path.join(args.out, "mini_trained.ply")
    export_splats(means=splats["means"], scales=splats["scales"],
                  quats=splats["quats"], opacities=splats["opacities"],
                  sh0=splats["sh0"], shN=splats["shN"],
                  format="ply", save_to=out_ply)
    # 保存归一化前相机位置（高度分层实验用）
    np.save(os.path.join(args.out, "raw_cam_pos.npy"),
            data["raw_cam_pos"].numpy())
    with open(os.path.join(args.out, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"PSNR={np.mean(psnrs):.4f}\nSSIM={np.mean(ssims):.4f}\n")
    print(f"模型已保存: {out_ply} （可拖进 SuperSplat）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--factor", type=int, default=8)
    ap.add_argument("--max_images", type=int, default=120)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--refine_stop", type=int, default=15000,
                    help="提前停止致密化可压低高斯数（8GB显存建议8000-12000）")
    ap.add_argument("--max_gaussians", type=int, default=0,
                    help=">0 时训练后按不透明度裁剪到该上限（如3000000）")
    ap.add_argument("--out", default="results/fuyao_mini")
    ap.add_argument("--seed", type=int, default=0, help="随机种子（多种子稳定性实验）")
    ap.add_argument("--all_train", action="store_true",
                    help="全量训练（不预留 val；held-out 评测交给独立架次数据）")
    ap.add_argument("--event_log", action="store_true",
                    help="记录致密化事件计数（ROI内外增量），输出 event_trace.csv")
    ap.add_argument("--h_radius_m", type=float, default=6.0,
                    help="ROI3D 物理半径（仅事件计数归属判定用，不影响 baseline 训练）")
    args = ap.parse_args()
    train(args)
