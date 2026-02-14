#!/usr/bin/env python
"""
CTransformer 分割结果可视化 — 生成 PNG 切片对比图。

用法:
  python scripts/visualize_seg.py \
    --seg-root scripts/results/ct_full_seg \
    --data-root /home/vm/workspace/DataSource/RunningDataset \
    --output scripts/results/seg_vis \
    --embryos 170704plc1p1 \
    --tp 50 100 150 \
    --slices 0.3 0.5 0.7

默认: 每个 embryo 取 3 个时间点, 每个时间点取 3 个 Z 切面
"""

import os
import glob
import argparse
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def random_label_cmap(n=256):
    """生成随机颜色表，0=黑色背景"""
    np.random.seed(42)
    colors = np.random.rand(n, 3)
    colors[0] = [0, 0, 0]  # 背景黑色
    return ListedColormap(colors)


def load_nii(path):
    return nib.load(path).get_fdata()


def normalize(arr):
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def get_available_tps(seg_root, embryo):
    """获取已有的时间点列表"""
    pattern = os.path.join(seg_root, embryo, 'SegCell', f'{embryo}_*_segCell.nii.gz')
    files = sorted(glob.glob(pattern))
    tps = []
    for f in files:
        tp_str = os.path.basename(f).split('_')[1]
        tps.append(int(tp_str))
    return tps


def visualize_one_tp(raw_memb, seg_memb, seg_cell, raw_nuc, z_fractions, embryo, tp, output_dir):
    """一个时间点: 生成一张 4行×N列 的对比图"""
    depth = raw_memb.shape[2]
    z_indices = [int(f * depth) for f in z_fractions]
    z_indices = [min(z, depth - 1) for z in z_indices]

    n_cols = len(z_indices)
    fig, axes = plt.subplots(4, n_cols, figsize=(5 * n_cols, 18))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    cmap_label = random_label_cmap()

    for col, z in enumerate(z_indices):
        # Row 0: Raw Membrane
        axes[0, col].imshow(normalize(raw_memb[:, :, z]), cmap='gray')
        axes[0, col].set_title(f'RawMemb Z={z}', fontsize=10)
        axes[0, col].axis('off')

        # Row 1: Raw Nucleus
        if raw_nuc is not None:
            axes[1, col].imshow(normalize(raw_nuc[:, :, z]), cmap='gray')
            axes[1, col].set_title(f'RawNuc Z={z}', fontsize=10)
        else:
            axes[1, col].set_title('RawNuc N/A', fontsize=10)
        axes[1, col].axis('off')

        # Row 2: Predicted Membrane (SegMemb)
        if seg_memb is not None:
            axes[2, col].imshow(normalize(seg_memb[:, :, z]), cmap='hot')
            axes[2, col].set_title(f'SegMemb Z={z}', fontsize=10)
        else:
            axes[2, col].set_title('SegMemb N/A', fontsize=10)
        axes[2, col].axis('off')

        # Row 3: Cell Instance Segmentation (SegCell)
        if seg_cell is not None:
            n_cells = len(np.unique(seg_cell[:, :, z])) - 1
            axes[3, col].imshow(seg_cell[:, :, z].astype(int) % 256, cmap=cmap_label,
                                interpolation='nearest')
            axes[3, col].set_title(f'SegCell Z={z} ({n_cells} cells)', fontsize=10)
        else:
            axes[3, col].set_title('SegCell N/A', fontsize=10)
        axes[3, col].axis('off')

    fig.suptitle(f'{embryo}  tp={tp}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{embryo}_tp{tp:03d}.png')
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  已保存: {save_path}')


def main():
    parser = argparse.ArgumentParser(description='CTransformer 分割可视化')
    parser.add_argument('--seg-root', default='scripts/results/ct_full_seg',
                        help='分割输出根目录')
    parser.add_argument('--data-root', default='/home/vm/workspace/DataSource/RunningDataset',
                        help='原始数据根目录')
    parser.add_argument('--output', default='scripts/results/seg_vis',
                        help='PNG 输出目录')
    parser.add_argument('--embryos', nargs='+', default=None,
                        help='指定 embryo (默认自动检测)')
    parser.add_argument('--tp', nargs='+', type=int, default=None,
                        help='指定时间点 (默认均匀取 3 个)')
    parser.add_argument('--slices', nargs='+', type=float, default=[0.3, 0.5, 0.7],
                        help='Z 轴切面比例 (默认 0.3 0.5 0.7)')
    args = parser.parse_args()

    # 自动检测 embryos
    if args.embryos:
        embryos = args.embryos
    else:
        embryos = sorted([
            d for d in os.listdir(args.seg_root)
            if os.path.isdir(os.path.join(args.seg_root, d, 'SegCell'))
        ])

    if not embryos:
        print('未找到分割结果，请检查 --seg-root')
        return

    print(f'可视化 {len(embryos)} 个胚胎, Z 切面比例: {args.slices}')

    for embryo in embryos:
        print(f'\n{"="*50}')
        print(f'  {embryo}')
        print(f'{"="*50}')

        available_tps = get_available_tps(args.seg_root, embryo)
        if not available_tps:
            print(f'  [跳过] 无 SegCell 文件')
            continue

        # 选择时间点
        if args.tp:
            tps = [t for t in args.tp if t in available_tps]
        else:
            # 均匀取 3 个
            n = len(available_tps)
            indices = [n // 4, n // 2, 3 * n // 4]
            tps = [available_tps[i] for i in indices]

        print(f'  共 {len(available_tps)} 个时间点, 可视化: {tps}')

        for tp in tps:
            tp_str = str(tp).zfill(3)

            # 加载分割结果
            seg_memb_path = os.path.join(args.seg_root, embryo, 'SegMemb',
                                         f'{embryo}_{tp_str}_segMemb.nii.gz')
            seg_cell_path = os.path.join(args.seg_root, embryo, 'SegCell',
                                         f'{embryo}_{tp_str}_segCell.nii.gz')
            raw_memb_path = os.path.join(args.data_root, embryo, 'RawMemb',
                                         f'{embryo}_{tp_str}_rawMemb.nii.gz')
            raw_nuc_path = os.path.join(args.data_root, embryo, 'RawNuc',
                                        f'{embryo}_{tp_str}_rawNuc.nii.gz')

            if not os.path.exists(raw_memb_path):
                print(f'  [跳过] tp={tp}: RawMemb 不存在')
                continue

            raw_memb = load_nii(raw_memb_path)
            raw_nuc = load_nii(raw_nuc_path) if os.path.exists(raw_nuc_path) else None
            seg_memb = load_nii(seg_memb_path) if os.path.exists(seg_memb_path) else None
            seg_cell = load_nii(seg_cell_path) if os.path.exists(seg_cell_path) else None

            visualize_one_tp(raw_memb, seg_memb, seg_cell, raw_nuc,
                             args.slices, embryo, tp, args.output)

    print(f'\n全部完成! PNG 输出: {args.output}')
    print(f'可用 scp 或 tar 传到本地查看:')
    print(f'  tar -czf seg_vis.tar.gz {args.output}')


if __name__ == '__main__':
    main()
