# TPS-3DGS（中文说明）

**面向垂直起降场标志小目标重建的定向光度监督（Targeted Photometric Supervision, TPS，3DGS）。**

3DGS 重建垂直起降场（VTOL）的着陆标志时，笔画边界会向外渗色、颜色被周边草地同化、笔画断裂。本文把这种失效追到 **光度监督稀释（PSD）**：图像级损失把每个残差都除以总像素数 `HW`，落到标志区域的归一化屏幕空间梯度只有 `0.04τ–0.32τ`，比致密化阈值低 3–25 倍，标志内的原语既不拟合也不细化。**TPS** 由停机坪标志的位置先验解析地投出随高度自适应的屏幕 ROI，只对 ROI 内像素的损失加权——渲染器、致密化判据、推理开销都不变。`w = 3` 时标志区 H-PSNR 提升 **+0.74 dB**（95% CI [+0.58, +0.90]，p < 0.0001，三个随机种子方向一致），全图 PSNR 变化 +0.03 dB（p = 0.63，不显著）。

本仓库公开：代码、单场景实拍飞行数据与稀疏重建、以及论文里所有结果表。训练好的 `.ply` 模型体积过大（单个 1–2 GB），未上传，可用下方命令重新生成。

## 内容

| 部分 | 路径 | 内容 |
|---|---|---|
| 代码 | `src/` | 训练（baseline / TPS / 各对照）、评测、出图 |
| 飞行数据 | `data/fieldH/` | 331 张 DJI Air 3S 照片（2048×1152）、COLMAP 文本模型、高度表 |
| 结果表 | `results/` | 各配置指标、统计检验、消融 |
| 文档 | `docs/` | 复现流程、结果说明 |

## 环境

已验证：Python 3.9 + PyTorch 2.8（CUDA 12.8）+ `gsplat` 1.5.3，单张 RTX 5070 Laptop（8 GB）。

```bash
pip install -r requirements.txt
```

`gsplat` 首次调用时会即时编译 CUDA 内核，因此需要 CUDA 工具链（`nvcc`）与构建工具：Linux 装 `ninja` 即可；Windows 需要 MSVC 生成工具并在 `vcvars64` 环境里运行，必要时手动设置 `TORCH_CUDA_ARCH_LIST`。

## 快速开始

```bash
# 0) 无 GPU 也跑得动的自检：数据加载、高度表、H 定位、ROI 投影
python src/selftest.py --data_dir data/fieldH --factor 8 --max_images 60

# 1) 标准 3DGS baseline
python src/mini_train.py --data_dir data/fieldH --factor 4 --max_images 331 \
    --max_steps 30000 --sh_degree 2 --refine_stop 12000 --out outputs/v4_fieldH/seed0_base

# 2) TPS（w = 3，只改损失权重，不动致密化判据）
python src/train_ours.py --data_dir data/fieldH --factor 4 --max_images 331 \
    --max_steps 30000 --sh_degree 2 --refine_stop 12000 \
    --loss_mode roi --h_roi_weight 3.0 --h_radius_m 6.0 --salience_strength 0 \
    --out outputs/v4_fieldH/seed0_ours

# 3) 评测与汇总
python src/eval_recon_v4.py --data_dir data/fieldH --root outputs/v4_fieldH --seeds 0 1 2
python src/aggregate_v4.py
```

配置与文件夹命名约定（`eval_recon_v4.py` 按 `<root>/seed<k>_<config>/mini_trained.ply` 找模型）：

| 文件夹后缀 | 参数 | 论文中的名称 |
|---|---|---|
| `base` | `mini_train.py` | 标准 baseline（w = 1） |
| `ours` | `--loss_mode roi --h_roi_weight 3 --salience_strength 0` | **TPS（w = 3）** |
| `w5` | `--h_roi_weight 5 --salience_strength 0` | 权重扫描 |
| `salonly` | `--h_roi_weight 1 --salience_strength 3` | 仅局部致密化（−0.25 dB） |
| `focal` | `--loss_mode focal` | Focal Loss（−1.42 dB） |
| `randroi` | `--loss_mode randroi` | 随机漂移 ROI（+0.23 dB） |
| `randfix` | `--loss_mode randfix --randfix_offset_m 25` | 固定偏移 ROI（+0.10 dB） |
| `gradscale` | `--loss_mode gradscale --h_roi_weight 3` | 梯度缩放（−0.14 dB） |

## 数据

`data/fieldH` 是 **2026 年 10 月 20 日于江西某垂直起降场** 用 DJI Air 3S 单架次环绕飞行拍摄的 331 张照片（草皮上的白色 “H” 标志），COLMAP 注册 329 张（未注册的两张是落地前后的低空帧），相对高度 21.9–93.7 m（绝对高度 23.4–118.0 m）。公开照片经过重编码到 2048×1152 并去除了 EXIF/XMP/GPS，因此随高度自适应的 ROI 需要的高度来自 `data/fieldH/altitude.csv`，`read_altitude.py` 会自动回落到这张表。

论文口径：每 8 张取 1 张做 held-out（41 视角），其余 288 张训练；41 个视角里 33 个能看到标志。H-PSNR 的评测区是以标志为中心、物方半径 4 m 的投影方框，另有 0–4 / 4–6 / 6–10 / 10–14 m 同心带。

未公开：训练好的 checkpoint（按 `docs/REPRODUCE.md` §2 自己练，单次约 23 分钟）。第二场景 `data/jxfx`（Table 10 用，117 张照片 + 位姿 + 高度表）也是公开的。

## 主要数字（三个种子，33 个 held-out 视角）

| 配置 | 全图 PSNR | 标志区 H-PSNR | ROI 内高斯数 | ΔH-PSNR |
|---|---|---|---|---|
| 标准 baseline（w = 1） | 24.24 ± 0.15 | 22.01 ± 0.25 | 6,700 ± 130 | — |
| **TPS（w = 3）** | 24.27 ± 0.04 | **22.74 ± 0.20** | 10,279 ± 556 | **+0.74** [+0.58, +0.90] |
| 仅局部致密化 | 24.15 ± 0.15 | 21.76 ± 0.08 | 15,975 ± 321 | −0.25 [−0.43, −0.07] |

补充证据：判据梯度全程落在 0.04τ–0.32τ；`w = 3` 时累积分裂资格从 167 升到 369，而 0–2 m 内圈分裂事件只占 0.42–0.75%；冻结拓扑后仅加权仍得 +0.272 dB；增益向内递增（0–4 m +1.16 dB、4–6 m +1.23 dB、6–10 m 接缝带 +0.83 dB、10–14 m +0.55 dB）；ROI 位置错开只剩 +0.23 / +0.10 dB。

训练成本（factor 4、30000 步）：约 23 分钟、峰值显存约 2.0 GB、约 199 万高斯。

## 许可与引用

代码 MIT（`LICENSE`）；`data/` 下照片与稀疏重建采用 CC BY 4.0，使用时请引用论文。机器可读条目见 `CITATION.cff`，BibTeX 见 `README.md`。
