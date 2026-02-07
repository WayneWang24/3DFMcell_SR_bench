#!/usr/bin/env python
"""
端到端 3D NIfTI 超分辨率脚本

输入: 原始 nii.gz 体积 (X, Y, Z)
输出: Z 轴超分后的 nii.gz (X, Y, Z*scale)

原理:
- 沿 Y 轴切成 XZ 切片，每张切片 shape = (Z, X)
- 对每张切片用 2D 模型超分，得到 (Z*scale, X)
- 堆叠回 3D 体积 (X, Y, Z*scale)

使用方法:
python scripts/eval/sr_3d_nifti.py \
  --input-dir /path/to/nii_volumes \
  --output-dir results/sr_3d/model_name \
  --config configs/edsr_x_cells_xy_template.py \
  --checkpoint results/work_dirs/edsr_x4/best_PSNR_iter_xxx.pth \
  --scale 4

批量多模型:
python scripts/eval/sr_3d_nifti.py \
  --input-dir /path/to/nii_volumes \
  --output-dir results/sr_3d \
  --config configs/edsr_x_cells_xy_template.py configs/srcnn_x_cells_xy_template.py \
  --checkpoint results/work_dirs/edsr_x4/best.pth results/work_dirs/srcnn_x4/best.pth \
  --scale 4
"""

import os
import argparse
import glob
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import cv2
from tqdm import tqdm
from mmengine.config import Config
from mmagic.registry import MODELS
# 兼容性修复：huggingface_hub 新版移除了 cached_download
import huggingface_hub
if not hasattr(huggingface_hub, 'cached_download'):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download
import mmagic.models  # noqa: F401  — 注册模型（需先修复 Adafactor 冲突，见 run_all_eval.sh）


def load_model(config_path, checkpoint_path, device='cuda'):
    """加载 mmagic 模型"""
    cfg = Config.fromfile(config_path)

    # 构建 generator
    model = MODELS.build(cfg.model.generator)

    # 加载权重
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    if 'state_dict' in ckpt:
        # 从完整 checkpoint 提取 generator 权重
        state_dict = {}
        for k, v in ckpt['state_dict'].items():
            if k.startswith('generator.'):
                state_dict[k.replace('generator.', '')] = v
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    return model


def normalize_slice(img, method='p1p99'):
    """归一化切片到 [0, 1]"""
    img = img.astype(np.float32)

    if method == 'p1p99':
        p1, p99 = np.percentile(img, [1, 99])
        img = np.clip(img, p1, p99)
        if p99 > p1:
            img = (img - p1) / (p99 - p1)
        else:
            img = img - p1
    elif method == 'minmax':
        vmin, vmax = img.min(), img.max()
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)
        else:
            img = img - vmin

    return img


def denormalize_slice(img, original_min, original_max):
    """反归一化回原始范围"""
    return img * (original_max - original_min) + original_min


def slice_to_tensor(img, device='cuda'):
    """
    2D numpy (H, W) -> torch tensor (1, 3, H, W)
    灰度复制到 3 通道
    """
    # 归一化到 [0, 1]
    if img.max() > 1.0:
        img = img / img.max()

    # 扩展到 3 通道
    img_3ch = np.stack([img, img, img], axis=0)  # (3, H, W)

    # 转 tensor
    tensor = torch.from_numpy(img_3ch).float().unsqueeze(0)  # (1, 3, H, W)

    return tensor.to(device)


def tensor_to_slice(tensor):
    """
    torch tensor (1, 3, H, W) -> 2D numpy (H, W)
    取第一个通道
    """
    img = tensor.squeeze(0).cpu().numpy()  # (3, H, W)
    img = img[0]  # 取第一个通道 (H, W)
    img = np.clip(img, 0, 1)
    return img


def super_resolve_slice(model, img, original_w=None, device='cuda'):
    """
    对单张 2D 切片做超分，只放大 H（Z 轴），W 保持原始尺寸。

    输入: (H, W) numpy array, 值域任意
    输出: (H*scale, W_original) numpy array, 值域 [0, 1]

    2D SR 模型会同时放大 H 和 W，所以需要在 W 方向 resize 回原始尺寸。
    """
    H, W = img.shape

    # 归一化
    img_norm = normalize_slice(img, 'p1p99')

    # 转 tensor
    tensor = slice_to_tensor(img_norm, device)

    # 推理
    with torch.no_grad():
        sr_tensor = model(tensor)

    # 转回 numpy
    sr_img = tensor_to_slice(sr_tensor)  # (H*scale, W*scale)

    # W 方向 resize 回原始尺寸（只保留 Z 轴的超分）
    target_w = original_w if original_w is not None else W
    if sr_img.shape[1] != target_w:
        sr_img = cv2.resize(sr_img, (target_w, sr_img.shape[0]),
                            interpolation=cv2.INTER_LINEAR)

    return sr_img


def process_volume_xz(vol, model, device='cuda', verbose=True):
    """
    沿 Y 轴处理体积（XZ 切片）

    输入体积: (X, Y, Z)
    每张切片: vol[:, y, :] -> (X, Z)
    超分后: (X, Z*scale)
    输出体积: (X, Y, Z*scale)
    """
    X, Y, Z = vol.shape

    # 先处理一张获取超分后的 Z 尺寸
    test_slice = vol[:, 0, :].T  # (Z, X)
    test_sr = super_resolve_slice(model, test_slice, original_w=X, device=device)
    Z_sr, _ = test_sr.shape

    # 创建输出体积
    vol_sr = np.zeros((X, Y, Z_sr), dtype=np.float32)

    # 逐切片处理
    iterator = range(Y)
    if verbose:
        iterator = tqdm(iterator, desc='  Processing XZ slices')

    for y in iterator:
        # 取 XZ 切片: (X, Z) -> 转置为 (Z, X) 作为图像
        xz_slice = vol[:, y, :].T  # (Z, X)

        # 超分（只放大 Z，W=X 保持不变）
        xz_sr = super_resolve_slice(model, xz_slice, original_w=X, device=device)  # (Z*scale, X)

        # 放回体积: (Z*scale, X) -> 转置为 (X, Z*scale)
        vol_sr[:, y, :] = xz_sr.T

    return vol_sr


def process_volume_yz(vol, model, device='cuda', verbose=True):
    """
    沿 X 轴处理体积（YZ 切片）

    输入体积: (X, Y, Z)
    每张切片: vol[x, :, :] -> (Y, Z)
    超分后: (Y, Z*scale)
    输出体积: (X, Y, Z*scale)
    """
    X, Y, Z = vol.shape

    # 先处理一张获取超分后的 Z 尺寸
    test_slice = vol[0, :, :].T  # (Z, Y)
    test_sr = super_resolve_slice(model, test_slice, original_w=Y, device=device)
    Z_sr, _ = test_sr.shape

    vol_sr = np.zeros((X, Y, Z_sr), dtype=np.float32)

    iterator = range(X)
    if verbose:
        iterator = tqdm(iterator, desc='  Processing YZ slices')

    for x in iterator:
        yz_slice = vol[x, :, :].T  # (Z, Y)
        yz_sr = super_resolve_slice(model, yz_slice, original_w=Y, device=device)  # (Z*scale, Y)
        vol_sr[x, :, :] = yz_sr.T

    return vol_sr


def save_nifti(vol, output_path, affine, scale=1):
    """保存为 NIfTI"""
    # 调整 Z 轴的 voxel size
    affine_adjusted = affine.copy()
    affine_adjusted[2, 2] = affine[2, 2] / scale

    # 转换为合适的数据类型
    # 输出是 [0, 1]，转为 16-bit
    vol_save = (vol * 65535).astype(np.uint16)

    img = nib.Nifti1Image(vol_save, affine_adjusted)
    nib.save(img, output_path)


def find_checkpoint(model_dir):
    """在模型目录中找到最佳 checkpoint"""
    model_dir = Path(model_dir)

    # 优先找 best
    best_ckpts = list(model_dir.glob('best_*.pth'))
    if best_ckpts:
        return str(best_ckpts[0])

    # 其次找最大的 iter
    iter_ckpts = list(model_dir.glob('iter_*.pth'))
    if iter_ckpts:
        iter_ckpts.sort(key=lambda x: int(x.stem.split('_')[-1]))
        return str(iter_ckpts[-1])

    # 最后找 last.pth
    last = model_dir / 'last.pth'
    if last.exists():
        return str(last)

    return None


def main():
    parser = argparse.ArgumentParser(description='端到端 3D NIfTI 超分辨率')
    parser.add_argument('--input-dir', required=True,
                        help='输入 nii.gz 目录')
    parser.add_argument('--output-dir', required=True,
                        help='输出目录')
    parser.add_argument('--config', nargs='+', required=True,
                        help='模型配置文件路径（支持多个）')
    parser.add_argument('--checkpoint', nargs='+', required=True,
                        help='模型 checkpoint 路径（支持多个，与 config 一一对应）')
    parser.add_argument('--scale', type=int, default=4,
                        help='超分倍率')
    parser.add_argument('--slice-axis', choices=['xz', 'yz'], default='xz',
                        help='切片方向: xz (沿 Y 切) 或 yz (沿 X 切)')
    parser.add_argument('--prefix', default='',
                        help='只处理此前缀的文件')
    parser.add_argument('--device', default='cuda',
                        help='计算设备')
    args = parser.parse_args()

    # 检查 config 和 checkpoint 数量匹配
    if len(args.config) != len(args.checkpoint):
        raise ValueError(f"config 数量 ({len(args.config)}) 与 checkpoint 数量 ({len(args.checkpoint)}) 不匹配")

    # 获取所有 nii.gz 文件（排除 nuc 开头的）
    input_files = sorted(glob.glob(os.path.join(args.input_dir, '*.nii.gz')))
    input_files = [f for f in input_files if not Path(f).name.startswith('nuc')]
    if args.prefix:
        input_files = [f for f in input_files if Path(f).name.startswith(args.prefix)]

    if not input_files:
        print(f"[错误] 未找到 nii.gz 文件: {args.input_dir}")
        return

    print(f"找到 {len(input_files)} 个 nii.gz 文件")
    print(f"处理 {len(args.config)} 个模型")
    print(f"切片方向: {args.slice_axis}")
    print(f"超分倍率: {args.scale}x")

    # 选择处理函数
    process_fn = process_volume_xz if args.slice_axis == 'xz' else process_volume_yz

    # 遍历模型
    for config_path, ckpt_path in zip(args.config, args.checkpoint):
        model_name = Path(ckpt_path).parent.name
        if model_name == '':
            model_name = Path(config_path).stem

        print(f"\n{'='*60}")
        print(f"模型: {model_name}")
        print(f"Config: {config_path}")
        print(f"Checkpoint: {ckpt_path}")

        # 加载模型
        print("加载模型...")
        try:
            model = load_model(config_path, ckpt_path, args.device)
        except Exception as e:
            print(f"[错误] 加载模型失败: {e}")
            continue

        # 创建输出目录
        model_output_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        # 处理每个文件
        for input_path in input_files:
            filename = Path(input_path).name
            print(f"\n处理: {filename}")

            # 读取
            try:
                nii = nib.load(input_path)
                vol = nii.get_fdata().astype(np.float32)
                affine = nii.affine
            except Exception as e:
                print(f"  [错误] 读取失败: {e}")
                continue

            print(f"  原始尺寸: {vol.shape}")

            # 超分
            try:
                vol_sr = process_fn(vol, model, args.device, verbose=True)
            except Exception as e:
                print(f"  [错误] 处理失败: {e}")
                continue

            print(f"  超分尺寸: {vol_sr.shape}")

            # 保存
            output_path = os.path.join(model_output_dir, filename.replace('.nii.gz', f'_sr_x{args.scale}.nii.gz'))
            save_nifti(vol_sr, output_path, affine, args.scale)
            print(f"  保存: {output_path}")

        # 清理显存
        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"完成！输出目录: {args.output_dir}")


if __name__ == '__main__':
    main()
