# --- open-source path shim (added for release) ---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import REPO_ROOT, DATA_ROOT, OUT_ROOT, RESULTS_ROOT  # noqa: E402
# -------------------------------------------------
# -*- coding: utf-8 -*-
"""train_ours.py —— 显著性预算 3DGS（本文方法，C2）

与 baseline（mini_train.py）完全同预算、同数据、同步数，唯一区别：
  1) 自动定位着陆标志 H 的 3D 位置；
  2) 每视角把 H 投影到图像，构造平滑 ROI 权重图；
  3) 对 ROI 像素做加权 L1（loss 放大→覆盖H的高斯反传梯度增大→
     DefaultStrategy 自然把致密化（尺度预算）向 H 倾斜）。
公平对比：相同 factor/SH/步数/refine_stop/高斯上限。

用法：
  python train_ours.py --data_dir data/fieldH --factor 4 --max_images 331 \
      --max_steps 30000 --sh_degree 2 --refine_stop 12000 \
      --max_gaussians 3000000 --h_roi_weight 3.0 --h_radius_m 6.0 \
      --out results/fieldH_ours
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from mini_train import (  # noqa: E402
    load_scene, init_splats, make_window, ssim, psnr,
)
from gsplat import rasterization, export_splats  # noqa: E402
from gsplat.strategy import DefaultStrategy  # noqa: E402
from gsplat.strategy.ops import duplicate, split  # noqa: E402
from read_altitude import read_xmp  # noqa: E402
from h3_util import infer_vax, metric_scale_from_T1  # noqa: E402


class SalienceStrategy(DefaultStrategy):
    """C2 致密化预算重分配：H 附近高斯更易分裂/克隆，其余区域等幅抑制。
    salience 权重均值归一化到 1，使总高斯预算保持守恒（与 baseline 公平对比）。"""

    def __init__(self, H3, vax, scale, radius_m=8.0, strength=3.0, **kw):
        super().__init__(**kw)
        self.H3 = np.asarray(H3, dtype=np.float64)
        self.vax = int(vax)
        self.scale = float(scale)
        self.hax = [a for a in range(3) if a != self.vax]
        self.r_norm = float(radius_m) / self.scale
        self.strength = float(strength)
        # 物理事件计数的多半径（m）：1.5/2 收紧到 H 笔画，3/4 标志外接框，6 预算圆柱
        self.evt_radii_m = [1.5, 2.0, 3.0, 4.0, 6.0]

    @torch.no_grad()
    def _salience(self, means):
        H3 = torch.as_tensor(self.H3, dtype=means.dtype, device=means.device)
        d0 = means[:, self.hax[0]] - H3[self.hax[0]]
        d1 = means[:, self.hax[1]] - H3[self.hax[1]]
        d = torch.sqrt(d0 * d0 + d1 * d1 + 1e-12)
        w = 1.0 + (self.strength - 1.0) * torch.exp(-(d ** 2) / (2.0 * self.r_norm ** 2))
        w = w * (w.numel() / w.sum().clamp_min(1e-6))  # 均值=1，预算守恒
        return w

    @torch.no_grad()
    def _grow_gs(self, params, optimizers, state, step):
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        grads = grads * self._salience(params["means"])  # ★显著区域致密化阈值降低
        device = grads.device
        is_grad_high = grads > self.grow_grad2d
        is_small = (torch.exp(params["scales"]).max(dim=-1).values
                    <= self.grow_scale3d * state["scene_scale"])
        is_dupli = is_grad_high & is_small
        n_dupli = is_dupli.sum().item()
        is_split = is_grad_high & (~is_small)
        if step < self.refine_scale2d_stop_iter:
            is_split |= state["radii"] > self.grow_scale2d
        n_split = is_split.sum().item()
        # —— 致密化事件 ROI 归属计数（事件计数 hook，供表 2 机制证据）——
        # 口径一：sal>1（显著性权重>1 的区域，仅 s>1 时非空；s=1 时处处为 1 恒为空）
        # 口径二：物理多半径圆柱（以 H3 为中心的水平距离；1.5/2m 收紧到笔画、3/4m 标志外接框、6m 预算圆柱）
        try:
            sal = self._salience(params["means"])
            self.last_dupli_roi = int((is_dupli & (sal > 1.0)).sum().item())
            self.last_split_roi = int((is_split & (sal > 1.0)).sum().item())
            _m = params["means"]
            _H3t = torch.as_tensor(self.H3, dtype=_m.dtype, device=_m.device)
            _d0 = _m[:, self.hax[0]] - _H3t[self.hax[0]]
            _d1 = _m[:, self.hax[1]] - _H3t[self.hax[1]]
            self._evt_dhor = torch.sqrt(_d0 * _d0 + _d1 * _d1 + 1e-12)
            self.dupli_phys, self.split_phys = [], []
            for _r in self.evt_radii_m:
                _inr = self._evt_dhor < (_r / self.scale)
                self.dupli_phys.append(int((is_dupli & _inr).sum().item()))
                self.split_phys.append(int((is_split & _inr).sum().item()))
            self.last_dupli_roi_phys = self.dupli_phys[-1]
            self.last_split_roi_phys = self.split_phys[-1]
        except Exception:
            self.last_dupli_roi = self.last_split_roi = 0
            self.last_dupli_roi_phys = self.last_split_roi_phys = 0
            self.dupli_phys = [0] * len(self.evt_radii_m)
            self.split_phys = [0] * len(self.evt_radii_m)
        if n_dupli > 0:
            duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)
        is_split = torch.cat([is_split, torch.zeros(n_dupli, dtype=torch.bool, device=device)])
        if n_split > 0:
            split(params=params, optimizers=optimizers, state=state, mask=is_split,
                  revised_opacity=self.revised_opacity)
        return n_dupli, n_split


class StatsStrategy(DefaultStrategy):
    """在每次致密化决策前，记录真实累积梯度的 ROI 统计（不改动决策）。"""

    def __init__(self, *a, roi_fn=None, logw=None, **kw):
        super().__init__(*a, **kw)
        self._roi_fn = roi_fn
        self._logw = logw

    @torch.no_grad()
    def _grow_gs(self, params, optimizers, state, step):
        try:
            grads = state["grad2d"] / state["count"].clamp_min(1)
            mask = self._roi_fn(params["means"])
            if int(mask.sum()) > 0:
                g = grads[mask].float()
                t = float(self.grow_grad2d)
                big = (torch.exp(params["scales"]).max(dim=-1).values
                       <= self.grow_scale3d * state["scene_scale"])[mask]
                hi = g > t
                self._logw.writerow([
                    step, int(mask.sum()), int(hi.sum()),
                    int((hi & ~big).sum()), int((hi & big).sum()),
                    float(g.median()) / t,
                    float(torch.quantile(g, 0.90)) / t,
                    float(torch.quantile(g, 0.99)) / t,
                    float(hi.float().mean()),
                    int(grads.numel()),
                ])
                self._logw_ok = True
        except Exception as e:  # 记录失败不应影响训练
            print("[stats warn]", e)
        return super()._grow_gs(params, optimizers, state, step)


def estimate_h3d(data, data_dir):
    """竖直轴由 XMP 真实高度相关性确定；H 水平位置=相机光心水平均值；
    竖直位置=H 水平邻域内 SfM 点云的地面高度（直方图主峰）。

    注意：不能用全局点云低分位数作地面——场景里的看台/树木/噪声等高于
    地面的点会把低百分位拉偏，使中心悬空约 7m，导致屏幕 ROI 投影整体
    偏移到 H 远侧（已用真实照片逐视角验证）。改用局部地面主峰。
    """
    cam = data["camtoworlds"][:, :3, 3].numpy()
    pts = data["points"].numpy()
    alt = []
    for nm in data["names"]:
        rel, _, _ = read_xmp(os.path.join(data_dir, "images", nm))
        alt.append(rel if rel is not None else np.nan)
    alt = np.array(alt)
    m = ~np.isnan(alt)
    rs = [abs(np.corrcoef(cam[m, a], alt[m])[0, 1]) for a in range(3)]
    vax = int(np.argmax(rs))
    A = np.polyfit(cam[m, vax], alt[m], 1)
    scale = abs(A[0])  # 1 归一化单位对应米数
    hax = [a for a in range(3) if a != vax]
    hx = float(cam[:, hax[0]].mean())
    hy = float(cam[:, hax[1]].mean())
    # 地面高度：H 水平邻域多个半径（4/6/8/10m）内点云沿竖直轴的
    # 直方图主峰，再取中位数，避免单一半径/bin 边界带来的抖动（约0.2m）
    d0 = pts[:, hax[0]] - hx
    d1 = pts[:, hax[1]] - hy
    dist = np.sqrt(d0 * d0 + d1 * d1)
    zgs = []
    for R in (4.0, 6.0, 8.0, 10.0):
        zv = pts[dist <= R / scale, vax]
        if len(zv) < 20:
            continue
        hist, edges = np.histogram(zv, bins=40)
        pk = edges[int(np.argmax(hist))]
        band = zv[(zv >= pk - 0.005) & (zv <= pk + 0.005)]
        zgs.append(float(np.median(band)) if len(band) else float(np.median(zv)))
    z_g = float(np.median(zgs))
    H3 = np.zeros(3, dtype=np.float64)
    H3[hax[0]] = hx
    H3[hax[1]] = hy
    H3[vax] = z_g
    print(f"  竖直轴=axis{vax} 1单位≈{scale:.1f}m；H3={H3}（局部地面z={z_g:.6f}，10m内点{int((dist<=10/scale).sum())}，"
          f"高度{np.polyval(A, z_g):.1f}m；旧全局5%分位会悬空）")
    return H3, vax, scale


def roi_weight_map(c2w, K, w, h, H3, scale, radius_m, w_hi, device,
                   roi_mode="3d", roi_px=90.0):
    """H 投影到当前视角，返回 h×w 平滑权重图（ROI 内 w_hi，其余 1）。

    roi_mode="3d"：固定三维物理半径，屏幕半径随观察高度(深度)自适应缩放（本文方法）；
    roi_mode="2d"：固定屏幕像素半径(roi_px)，不随高度缩放（静态对照，用于机理消融）。
    两种模式下 ROI 中心均由 H3 投影给出，唯一差别是屏幕半径是否随高度自适应。
    """
    viewmat = torch.linalg.inv(c2w)[0].detach().cpu().numpy()
    pc = viewmat @ np.array([H3[0], H3[1], H3[2], 1.0])
    W = torch.ones((h, w), device=device)
    if pc[2] <= 0:
        return W
    Kn = K[0].detach().cpu().numpy()
    q = Kn @ pc[:3]
    u, v = q[0] / q[2], q[1] / q[2]
    if roi_mode == "2d":
        rpx = float(roi_px)  # 固定屏幕半径，不随深度缩放
    else:
        r_norm = radius_m / scale
        rpx = Kn[0, 0] * r_norm / pc[2]  # 近大远小，随高度自适应
    if rpx < 1 or u < -rpx or u > w + rpx or v < -rpx or v > h + rpx:
        return W
    yy, xx = torch.meshgrid(torch.arange(h, device=device).float(),
                            torch.arange(w, device=device).float(), indexing="ij")
    d2 = (xx - u) ** 2 + (yy - v) ** 2
    sigma = max(rpx, 1.0)
    W = 1.0 + (w_hi - 1.0) * torch.exp(-d2 / (2.0 * sigma ** 2))
    return W


def roi3d_mask(means, H3, hax, r_norm):
    """3D 物理距离判定高斯是否落在 H 标志 ROI 内（事件计数用，与策略无关）。"""
    d0 = means[:, hax[0]] - H3[hax[0]]
    d1 = means[:, hax[1]] - H3[hax[1]]
    d = torch.sqrt(d0 * d0 + d1 * d1)
    return d <= r_norm


def train(args):
    device = "cuda"
    print(f"[1/5] 加载场景: {args.data_dir}")
    data = load_scene(args.data_dir, args.factor, args.max_images)
    n = len(data["pixels"])
    if args.all_train:
        val_ids = set()
        train_ids = list(range(n))
        print(f"  共 {n} 视角：全量训练（held-out 评测交由独立架次数据）")
    else:
        val_ids = set(range(7, n, 8))
        train_ids = [i for i in range(n) if i not in val_ids]
        print(f"  共 {n} 视角：训练 {len(train_ids)}，测试 {len(val_ids)}")
    if args.h3_override:
        h3o = np.array([float(x) for x in args.h3_override.split(",")],
                       dtype=np.float64)
        assert len(h3o) == 3, "h3_override 需为 'x,y,z' 三个归一化坐标"
        H3 = h3o
        vax = infer_vax(data["camtoworlds"].numpy())
        scale = float(args.scale_m) if float(args.scale_m) > 0 else metric_scale_from_T1(data["T1"])
        print(f"  [h3_override] 使用外部 H3={H3}（vax=axis{vax}, scale={scale:.3f} m/unit，"
              f"不读 XMP）")
    else:
        H3, vax, scale = estimate_h3d(data, args.data_dir)
    hax = [a for a in range(3) if a != vax]

    # randfix：固定一个偏离 H 的三维错误位置（跨所有视角一致，由 seed 可复现）。
    # 与 randroi（每步随机漂移、等价随机重加权正则化）不同，randfix 检验"固定位置先验是否必须对准 H"。
    H3_bad = None
    if args.loss_mode == "randfix":
        _rng_fix = np.random.RandomState(args.seed + 999)
        _ang = _rng_fix.uniform(0.0, 2.0 * np.pi)
        _off_m = float(getattr(args, "randfix_offset_m", 25.0))
        H3_bad = H3.copy()
        H3_bad[hax[0]] += (_off_m / scale) * float(np.cos(_ang))
        H3_bad[hax[1]] += (_off_m / scale) * float(np.sin(_ang))
        print(f"  [randfix] 固定错误位置：相对 H 水平偏移 {_off_m:.1f} m，"
              f"方位角 {_ang:.2f} rad（SfM 单位 {_off_m/scale:.5f}）")

    print("[2/5] 初始化高斯 + 致密化策略")
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"  随机种子 seed={args.seed}（控制初始四元数、视角采样、致密化随机）")
    if args.init_ply:
        from eval_altitude import load_ply_splats
        _sp = load_ply_splats(args.init_ply, args.sh_degree, device)
        splats = {k: torch.nn.Parameter(v.detach().clone().requires_grad_(True))
                  for k, v in _sp.items()}
        print(f"  [init] 从 PLY 起训: {args.init_ply}  N={len(splats['means'])}")
    else:
        splats = init_splats(data["points"], data["points_rgb"], args.sh_degree, device)
    opt = {
        k: torch.optim.Adam([{"params": splats[k], "lr": lr}], eps=1e-15)
        for k, lr in zip(
            ["means", "scales", "quats", "opacities", "sh0", "shN"],
            [1.6e-4, 5e-3, 1e-3, 5e-2, 2.5e-3, 2.5e-3 / 20],
        )
    }
    if args.salience_strength > 0:
        strategy = SalienceStrategy(
            H3, vax, scale, radius_m=args.salience_radius_m,
            strength=args.salience_strength,
            verbose=False, refine_stop_iter=args.refine_stop)
        print(f"  [C2] 显著性致密化预算重分配：H半径 {args.salience_radius_m}m，强度 {args.salience_strength}")
    else:
        strategy = DefaultStrategy(verbose=False, refine_stop_iter=args.refine_stop)
    if args.grad_stats:
        os.makedirs(args.out, exist_ok=True);         import csv as _csv3
        _sf = open(os.path.join(args.out, "grad_accum.csv"), "w", newline="")
        _sw = _csv3.writer(_sf)
        _sw.writerow(["step", "n_roi", "n_over_tau", "n_over_split_elig",
                      "n_over_clone_elig", "q50_x_tau", "q90_x_tau",
                      "q99_x_tau", "frac_over_tau", "n_all"])
        _hax_t = [a for a in range(3) if a != vax]
        _rn_t = args.h_radius_m / scale
        _roi_fn = (lambda m: roi3d_mask(m, H3, _hax_t, _rn_t))
        if isinstance(strategy, DefaultStrategy):
            _kw = dict(verbose=False, refine_stop_iter=args.refine_stop)
            strategy = StatsStrategy(roi_fn=_roi_fn, logw=_sw, **_kw)
    if args.freeze_densify:
        strategy.refine_stop_iter = 0
        print("  [freeze] 克隆/分裂/剪枝全部冻结（拓扑固定）")
    strategy.check_sanity(splats, opt)
    state = strategy.initialize_state(scene_scale=1.0)
    window = make_window(device)
    means_sched = torch.optim.lr_scheduler.ExponentialLR(
        opt["means"], gamma=0.01 ** (1.0 / args.max_steps))
    os.makedirs(args.out, exist_ok=True)

    def save_ply(path, sp):
        export_splats(means=sp["means"], scales=sp["scales"], quats=sp["quats"],
                      opacities=sp["opacities"], sh0=sp["sh0"], shN=sp["shN"],
                      format="ply", save_to=path)

    print(f"[3/5] 训练 {args.max_steps} 步（loss_mode={args.loss_mode}, w_hi={args.h_roi_weight}, r={args.h_radius_m}m）")
    if args.grad_hook:
        import csv as _csv
        _gtf = open(os.path.join(args.out, "grad_trace.csv"), "w", newline="")
        _gtw = _csv.writer(_gtf)
        _gtw.writerow(["step", "n_gauss", "n_roi", "grad_roi", "grad_out", "delta_splits"])
        H3t = torch.tensor(H3, dtype=torch.float32, device=device)
    if args.event_log:
        import csv as _csv2
        _ef = open(os.path.join(args.out, "event_trace.csv"), "w", newline="")
        _ew = _csv2.writer(_ef)
        _ew.writerow(["step", "n_gauss", "n_roi3d", "delta_gauss",
                      "delta_roi3d", "clone_roi", "split_roi",
                      "clone_roi_phys", "split_roi_phys",
                      "n_r15", "n_r20", "n_r30", "n_r40",
                      "clo_r15", "clo_r20", "clo_r30", "clo_r40", "clo_r60",
                      "spl_r15", "spl_r20", "spl_r30", "spl_r40", "spl_r60",
                      "opac_roi_mean", "opac_out_mean"])
        _r_norm = args.h_radius_m / scale
        _hax = hax

    t0 = time.time()
    for step in range(args.max_steps):
        idx = train_ids[np.random.randint(len(train_ids))]
        gt = data["pixels"][idx].to(device)
        w, h = data["wh"][idx]
        c2w = data["camtoworlds"][idx:idx + 1].to(device)
        K = data["Ks"][idx:idx + 1].to(device)

        if args.grad_hook:
            n_before = len(splats["means"])
        if args.event_log:
            n_eb = len(splats["means"])
            with torch.no_grad():
                _rm_b = roi3d_mask(splats["means"], H3, _hax, _r_norm)
                n_roi_b = int(_rm_b.sum().item())

        colors = torch.cat([splats["sh0"], splats["shN"]], 1)
        renders, alphas, info = rasterization(
            means=splats["means"], quats=splats["quats"],
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=colors, viewmats=torch.linalg.inv(c2w), Ks=K,
            width=w, height=h, sh_degree=min(step // 1000, args.sh_degree),
            absgrad=True, packed=False,
        )
        pred = renders[0].clamp(0, 1)
        diff = (pred - gt).abs().mean(dim=2)
        if args.loss_mode == "roi":
            # —— 唯一与 baseline 不同：H ROI 加权 L1 ——
            W = roi_weight_map(c2w, K, w, h, H3, scale,
                               args.h_radius_m, args.h_roi_weight, device,
                               roi_mode=args.roi_mode, roi_px=args.roi_px)
            l1 = 0.8 * (W * diff).sum() / W.sum()
        elif args.loss_mode == "randroi":
            # 随机同面积 ROI：中心/半径随机，验证"位置先验"是否为增益来源
            import random as _random
            u = _random.uniform(0, w); v = _random.uniform(0, h)
            rpx = max(K[0, 0, 0].item() * (args.h_radius_m / scale) / 60.0, 20.0)
            yy, xx = torch.meshgrid(torch.arange(h, device=device).float(),
                                    torch.arange(w, device=device).float(), indexing="ij")
            d2 = (xx - u) ** 2 + (yy - v) ** 2
            W = 1.0 + (args.h_roi_weight - 1.0) * torch.exp(-d2 / (2.0 * (max(rpx, 1.0)) ** 2))
            l1 = 0.8 * (W * diff).sum() / W.sum()
        elif args.loss_mode == "randfix":
            # 固定错误位置 ROI：与真实 H 完全相同的 3D 高度自适应投影/半径/权重，
            # 仅中心换成偏离 H 的固定点（跨视角一致）。检验位置先验是否必须对准 H。
            W = roi_weight_map(c2w, K, w, h, H3_bad, scale,
                               args.h_radius_m, args.h_roi_weight, device,
                               roi_mode="3d")
            l1 = 0.8 * (W * diff).sum() / W.sum()
        elif args.loss_mode == "focal":
            # 难样本加权（与空间位置无关）：p 高置信度处降权
            p = torch.exp(-diff)  # 像素相似度
            W = (1 - p) ** 2
            W = W / W.mean()
            l1 = 0.8 * (W * diff).sum() / W.sum()
        else:  # gradscale：像素级 loss 无权重（标准 L1），仅 backward 后放大 ROI 高斯梯度
            l1 = 0.8 * diff.mean()
        l_ssim = 0.2 * (1 - ssim(pred.permute(2, 0, 1)[None],
                                 gt.permute(2, 0, 1)[None], window).mean())
        loss = l1 + l_ssim

        strategy.step_pre_backward(params=splats, optimizers=opt,
                                   state=state, step=step, info=info)
        loss.backward()
        if args.loss_mode == "gradscale":
            # 梯度缩放：直接放大 ROI 内高斯的位置梯度（不改变像素级损失权重）
            with torch.no_grad():
                rm = roi3d_mask(splats["means"], H3, hax, args.h_radius_m / scale)
                if rm.any():
                    splats["means"].grad[rm] *= args.h_roi_weight
        if args.grad_hook and step % args.grad_hook_every == 0:
            with torch.no_grad():
                g = splats["means"].grad.norm(dim=-1)
                viewmat = torch.linalg.inv(c2w)[0]
                cam = viewmat[:3, :3] @ splats["means"].T + viewmat[:3, 3:4]
                depth = cam[2].clamp_min(1e-6)
                Kk = K[0]
                u = Kk[0, 0] * cam[0] / depth + Kk[0, 2]
                v = Kk[1, 1] * cam[1] / depth + Kk[1, 2]
                pc = viewmat @ torch.cat([H3t, torch.ones(1, device=device)])
                if pc[2] > 0:
                    u0 = Kk[0, 0] * pc[0] / pc[2] + Kk[0, 2]
                    v0 = Kk[1, 1] * pc[1] / pc[2] + Kk[1, 2]
                    if args.roi_mode == "2d":
                        rpx = args.roi_px
                    else:
                        rpx = Kk[0, 0] * (args.h_radius_m / scale) / pc[2]
                    d2 = (u - u0) ** 2 + (v - v0) ** 2
                    in_roi = (depth > 0) & (d2 <= (max(rpx, 1.0) * 1.5) ** 2)
                    _n_roi = int(in_roi.sum().item())
                    _g_roi = float(g[in_roi].mean().item()) if _n_roi else float("nan")
                    _g_out = float(g[~in_roi].mean().item()) if (~in_roi).any() else float("nan")
                else:
                    _n_roi, _g_roi, _g_out = 0, float("nan"), float("nan")
        for o in opt.values():
            o.step(); o.zero_grad(set_to_none=True)
        means_sched.step()
        strategy.step_post_backward(params=splats, optimizers=opt, state=state,
                                    step=step, info=info, packed=False)

        if args.grad_hook and step % args.grad_hook_every == 0:
            _gtw.writerow([step, len(splats["means"]), _n_roi,
                           f"{_g_roi:.5f}", f"{_g_out:.5f}", len(splats["means"]) - n_before])
            _gtf.flush()

        if args.event_log and step % 100 == 0:
            with torch.no_grad():
                _rm_a = roi3d_mask(splats["means"], H3, _hax, _r_norm)
                n_roi_a = int(_rm_a.sum().item())
                opa = torch.sigmoid(splats["opacities"]).squeeze()
                opac_roi = float(opa[_rm_a].mean().item()) if n_roi_a else float("nan")
                opac_out = float(opa[~_rm_a].mean().item()) if (~_rm_a).any() else float("nan")
                _clo = getattr(strategy, "last_dupli_roi", 0)
                _spl = getattr(strategy, "last_split_roi", 0)
                _clo_p = getattr(strategy, "last_dupli_roi_phys", 0)
                _spl_p = getattr(strategy, "last_split_roi_phys", 0)
                # 多物理半径（1.5/2/3/4/6 m）高斯数与 clone/split 事件，区分笔画与外围地面
                _mmc = splats["means"]
                _H3c = torch.as_tensor(H3, dtype=_mmc.dtype, device=_mmc.device)
                _d0c = _mmc[:, hax[0]] - _H3c[hax[0]]
                _d1c = _mmc[:, hax[1]] - _H3c[hax[1]]
                _dhorc = torch.sqrt(_d0c * _d0c + _d1c * _d1c + 1e-12)
                _er = [1.5, 2.0, 3.0, 4.0, 6.0]
                _nrs = [int((_dhorc < (_r / scale)).sum().item()) for _r in _er]
                _clos = getattr(strategy, "dupli_phys", [0] * 5)
                _spls = getattr(strategy, "split_phys", [0] * 5)
                _ew.writerow([step, len(splats["means"]), n_roi_a,
                              len(splats["means"]) - n_eb, n_roi_a - n_roi_b,
                              _clo, _spl, _clo_p, _spl_p,
                              _nrs[0], _nrs[1], _nrs[2], _nrs[3],
                              _clos[0], _clos[1], _clos[2], _clos[3], _clos[4],
                              _spls[0], _spls[1], _spls[2], _spls[3], _spls[4],
                              f"{opac_roi:.4f}", f"{opac_out:.4f}"])
                _ef.flush()
            n_eb = len(splats["means"])

        if step % 250 == 0 or step == args.max_steps - 1:
            mem = torch.cuda.max_memory_allocated() / 1024 ** 3
            print(f"  step {step:5d} | loss {loss.item():.4f} | PSNR {psnr(pred,gt).item():.2f} "
                  f"| 高斯 {len(splats['means']):,} | 显存 {mem:.1f}GB | {step/max(time.time()-t0,1):.1f} it/s")
        if step > 0 and step % 5000 == 0:
            save_ply(os.path.join(args.out, f"ckpt_{step}.ply"), splats)
    if args.grad_hook:
        _gtf.close()

    if args.max_gaussians > 0 and len(splats["means"]) > args.max_gaussians:
        with torch.no_grad():
            opa = torch.sigmoid(splats["opacities"]).squeeze()
            keep = torch.topk(opa, args.max_gaussians).indices.sort().values
        for k in splats:
            splats[k] = torch.nn.Parameter(splats[k].data[keep])
        print(f"  按不透明度裁剪到 {args.max_gaussians:,} 高斯")
    print("[4/5] 全图测试集评测")
    psnrs, ssims = [], []
    if val_ids:
        with torch.no_grad():
            for idx in sorted(val_ids):
                gt = data["pixels"][idx].to(device)
                w, h = data["wh"][idx]
                c2w = data["camtoworlds"][idx:idx + 1].to(device)
                K = data["Ks"][idx:idx + 1].to(device)
                renders, _, _ = rasterization(
                    means=splats["means"], quats=splats["quats"],
                    scales=torch.exp(splats["scales"]),
                    opacities=torch.sigmoid(splats["opacities"]),
                    colors=torch.cat([splats["sh0"], splats["shN"]], 1),
                    viewmats=torch.linalg.inv(c2w), Ks=K,
                    width=w, height=h, sh_degree=args.sh_degree)
                pred = renders[0].clamp(0, 1)
                psnrs.append(psnr(pred, gt).item())
                ssims.append(ssim(pred.permute(2, 0, 1)[None],
                                  gt.permute(2, 0, 1)[None], window).mean().item())
        print(f"  全图测试 PSNR = {np.mean(psnrs):.2f} dB / SSIM = {np.mean(ssims):.4f}")
    else:
        print("  全量训练模式：本模型不做同架次评测（held-out 评测由独立架次数据完成）")

    print("[5/5] 保存模型")
    out_ply = os.path.join(args.out, "mini_trained.ply")
    export_splats(means=splats["means"], scales=splats["scales"], quats=splats["quats"],
                  opacities=splats["opacities"], sh0=splats["sh0"], shN=splats["shN"],
                  format="ply", save_to=out_ply)
    np.save(os.path.join(args.out, "raw_cam_pos.npy"), data["raw_cam_pos"].numpy())
    with open(os.path.join(args.out, "metrics.txt"), "w", encoding="utf-8") as f:
        if psnrs:
            f.write(f"PSNR={np.mean(psnrs):.4f}\nSSIM={np.mean(ssims):.4f}\n")
        else:
            f.write("PSNR=N/A (all_train, held-out eval via separate sortie)\n")
    print("OURS_TRAIN_DONE:", out_ply, f"({len(splats['means']):,} 高斯)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--max_images", type=int, default=331)
    ap.add_argument("--max_steps", type=int, default=30000)
    ap.add_argument("--sh_degree", type=int, default=2)
    ap.add_argument("--refine_stop", type=int, default=12000)
    ap.add_argument("--max_gaussians", type=int, default=3000000)
    ap.add_argument("--h_roi_weight", type=float, default=3.0)
    ap.add_argument("--h_radius_m", type=float, default=6.0)
    ap.add_argument("--roi_mode", choices=["3d", "2d"], default="3d",
                    help="ROI屏幕半径：3d=随高度自适应的物理半径(本文)，2d=固定屏幕像素(静态对照)")
    ap.add_argument("--roi_px", type=float, default=90.0,
                    help="roi_mode=2d 时的固定屏幕像素半径")
    ap.add_argument("--init_ply", default="")
    ap.add_argument("--freeze_densify", action="store_true")
    ap.add_argument("--grad_stats", action="store_true")
    ap.add_argument("--grad_hook", action="store_true",
                    help="记录ROI内外位置梯度均值与分裂增量，输出 grad_trace.csv（因果证据，不改变训练行为）")
    ap.add_argument("--grad_hook_every", type=int, default=250)
    ap.add_argument("--salience_strength", type=float, default=3.0,
                    help="H区域致密化强度(>1)；0=只保留ROI加权、不做预算重分配")
    ap.add_argument("--salience_radius_m", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0, help="随机种子（多种子稳定性实验）")
    ap.add_argument("--all_train", action="store_true",
                    help="全量训练（不预留 val；held-out 评测交给独立架次数据）")
    ap.add_argument("--loss_mode", choices=["roi", "randroi", "randfix", "focal", "gradscale"], default="roi",
                    help="roi=本文ROI加权L1；randroi=每步随机漂移ROI；randfix=固定偏离H的错误位置ROI（跨视角一致，检验位置先验特异性）；focal=难样本加权；gradscale=ROI高斯梯度缩放（像素级loss无权重）")
    ap.add_argument("--event_log", action="store_true",
                    help="记录致密化事件计数（ROI内外增量、clone/split归属、不透明度加权密度），输出 event_trace.csv")
    ap.add_argument("--budget_multiplier", type=float, default=1.0,
                    help="高斯上限放大系数（budget 强基线：>1 表示直接增加预算）")
    ap.add_argument("--randfix_offset_m", type=float, default=25.0,
                    help="randfix 固定错误位置相对 H 的水平偏移（米）")
    ap.add_argument("--h3_override", default="",
                    help="外部 H3 归一化坐标 'x,y,z'，覆盖 estimate_h3d 的水平/竖直定位；"
                         "vax/scale 仍自动（用于第二场景：全量光心均值会被高空大圈带偏）")
    ap.add_argument("--scale_m", type=float, default=0.0,
                    help="公开数据可选：1 归一化单位对应的米数；<=0 时从 T1 推断")
    ap.add_argument("--out", default="results/fieldH_ours")
    args = ap.parse_args()
    if args.budget_multiplier != 1.0:
        args.max_gaussians = int(args.max_gaussians * args.budget_multiplier)
    train(args)
