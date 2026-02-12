#!/usr/bin/env python
"""
将原始数据和 SR 输出转换为 CTransformer 期望的目录结构。

CTransformer 期望:
  DataSource/RunningDataset/{embryo}/RawMemb/{embryo}_{tp}_rawMemb.nii.gz
  DataSource/RunningDataset/{embryo}/RawNuc/{embryo}_{tp}_rawNuc.nii.gz

本脚本:
1. 原始数据: symlink memb + nuc (尺寸一致，直接链接)
2. SR 数据: symlink SR memb + 将 nuc 沿 Z 插值到 SR 尺寸

使用方法:
  # fastz + XZ 方向 SR
  python scripts/prepare_ct_data.py \
    --raw-dir data/raw_cell_datasets/fastz_volumes \
    --sr-dir scripts/results/eval_outputs/sr_3d_fastz_xz \
    --output-dir CTransformer/DataSource/RunningDataset \
    --condition fastz

  # single_shot + XZ 方向 SR
  python scripts/prepare_ct_data.py \
    --raw-dir data/raw_cell_datasets/single_shot_volumes \
    --sr-dir scripts/results/eval_outputs/sr_3d_singleS_xz \
    --output-dir CTransformer/DataSource/RunningDataset \
    --condition singleS
"""

import os
import sys
import glob
import argparse
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


def parse_series_name(filename):
    """
    从文件名中提取序号作为时间点编号。

    memb_Series001.nii.gz       -> 001
    nuc_Series001.nii.gz        -> 001
    memb_L_1.nii.gz             -> 001
    memb_Series001_sr_x4.nii.gz -> 001
    """
    stem = filename.replace('.nii.gz', '')
    # 去掉 SR 后缀
    for suffix in ['_sr_x2', '_sr_x4', '_sr_x8']:
        stem = stem.replace(suffix, '')

    if 'Series' in stem:
        num = stem.split('Series')[-1]
        return num.zfill(3)

    if '_L_' in stem:
        num = stem.split('_L_')[-1]
        return str(int(num)).zfill(3)

    return stem.split('_')[-1].zfill(3)


def find_nuc_file(raw_dir, memb_filename):
    """根据 memb 文件名找到对应的 nuc 文件"""
    nuc_name = memb_filename.replace('memb_', 'nuc_')
    # 去掉 SR 后缀
    for suffix in ['_sr_x2', '_sr_x4', '_sr_x8']:
        nuc_name = nuc_name.replace(suffix, '')

    nuc_path = os.path.join(raw_dir, nuc_name)
    if os.path.exists(nuc_path):
        return nuc_path
    return None


def resample_nuc_z(nuc_path, target_z, output_path):
    """将 nuc 体积沿 Z 轴插值到目标尺寸"""
    if os.path.exists(output_path):
        return

    nii = nib.load(nuc_path)
    data = nii.get_fdata()
    affine = nii.affine.copy()

    # data shape: (X, Y, Z_orig), target: (X, Y, target_z)
    z_orig = data.shape[2]
    if z_orig == target_z:
        # 尺寸一致，直接拷贝
        shutil.copy2(nuc_path, output_path)
        return

    z_factor = target_z / z_orig
    resampled = zoom(data, (1, 1, z_factor), order=1)  # 线性插值

    # 更新 affine 的 Z 轴 voxel size
    affine[2, 2] = affine[2, 2] / z_factor

    out_nii = nib.Nifti1Image(resampled.astype(data.dtype), affine)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(out_nii, output_path)


def prepare_original(raw_dir, embryo_name, output_root, use_symlink=True):
    """
    原始数据: memb 和 nuc 尺寸一致，直接 symlink。
    """
    memb_files = sorted(glob.glob(os.path.join(raw_dir, 'memb_*.nii.gz')))
    memb_out_dir = os.path.join(output_root, embryo_name, 'RawMemb')
    nuc_out_dir = os.path.join(output_root, embryo_name, 'RawNuc')
    os.makedirs(memb_out_dir, exist_ok=True)
    os.makedirs(nuc_out_dir, exist_ok=True)

    count = 0
    for memb_path in memb_files:
        filename = os.path.basename(memb_path)
        tp = parse_series_name(filename)

        target_memb = os.path.join(memb_out_dir, f'{embryo_name}_{tp}_rawMemb.nii.gz')
        target_nuc = os.path.join(nuc_out_dir, f'{embryo_name}_{tp}_rawNuc.nii.gz')

        # memb: symlink
        if not os.path.exists(target_memb):
            src = os.path.abspath(memb_path)
            if use_symlink:
                os.symlink(src, target_memb)
            else:
                shutil.copy2(src, target_memb)

        # nuc: symlink (尺寸一致)
        nuc_path = find_nuc_file(raw_dir, filename)
        if nuc_path and not os.path.exists(target_nuc):
            src = os.path.abspath(nuc_path)
            if use_symlink:
                os.symlink(src, target_nuc)
            else:
                shutil.copy2(src, target_nuc)

        count += 1
    return count


def prepare_sr(sr_model_dir, raw_dir, embryo_name, output_root, use_symlink=True):
    """
    SR 数据: memb 用 SR 输出，nuc 从原始插值到 SR 的 Z 尺寸。
    """
    sr_files = sorted(glob.glob(os.path.join(sr_model_dir, '*.nii.gz')))
    if not sr_files:
        return 0

    memb_out_dir = os.path.join(output_root, embryo_name, 'RawMemb')
    nuc_out_dir = os.path.join(output_root, embryo_name, 'RawNuc')
    os.makedirs(memb_out_dir, exist_ok=True)
    os.makedirs(nuc_out_dir, exist_ok=True)

    count = 0
    for sr_path in sr_files:
        filename = os.path.basename(sr_path)
        tp = parse_series_name(filename)

        target_memb = os.path.join(memb_out_dir, f'{embryo_name}_{tp}_rawMemb.nii.gz')
        target_nuc = os.path.join(nuc_out_dir, f'{embryo_name}_{tp}_rawNuc.nii.gz')

        # memb: symlink SR 输出
        if not os.path.exists(target_memb):
            src = os.path.abspath(sr_path)
            if use_symlink:
                os.symlink(src, target_memb)
            else:
                shutil.copy2(src, target_memb)

        # nuc: 插值到 SR memb 的 Z 尺寸
        nuc_path = find_nuc_file(raw_dir, filename)
        if nuc_path and not os.path.exists(target_nuc):
            sr_nii = nib.load(sr_path)
            target_z = sr_nii.shape[2]
            resample_nuc_z(nuc_path, target_z, target_nuc)

        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description='准备 CTransformer 输入数据')
    parser.add_argument('--raw-dir', required=True,
                        help='原始数据目录 (包含 memb_*.nii.gz 和 nuc_*.nii.gz)')
    parser.add_argument('--sr-dir', required=True,
                        help='SR 输出目录 (包含各模型子目录)')
    parser.add_argument('--output-dir', required=True,
                        help='CTransformer DataSource/RunningDataset 路径')
    parser.add_argument('--condition', required=True,
                        help='条件名称 (fastz / singleS)')
    parser.add_argument('--copy', action='store_true',
                        help='拷贝文件而不是创建 symlink')
    args = parser.parse_args()

    use_symlink = not args.copy

    # ============================================================
    # 1. 原始数据
    # ============================================================
    embryo_orig = f'{args.condition}_original'
    n = prepare_original(args.raw_dir, embryo_orig, args.output_dir, use_symlink)
    print(f'[原始数据] {embryo_orig}: {n} 个时间点')

    # ============================================================
    # 2. 各 SR 模型
    # ============================================================
    if not os.path.exists(args.sr_dir):
        print(f"[警告] SR 目录不存在: {args.sr_dir}")
        return

    model_dirs = sorted([
        d for d in os.listdir(args.sr_dir)
        if os.path.isdir(os.path.join(args.sr_dir, d))
    ])

    for model_name in model_dirs:
        model_path = os.path.join(args.sr_dir, model_name)
        embryo_sr = f'{args.condition}_{model_name}'
        n = prepare_sr(model_path, args.raw_dir, embryo_sr, args.output_dir, use_symlink)
        if n > 0:
            print(f'[SR 模型] {embryo_sr}: {n} 个时间点 (nuc 已插值)')

    # ============================================================
    # 3. 汇总
    # ============================================================
    all_embryos = sorted([
        d for d in os.listdir(args.output_dir)
        if os.path.isdir(os.path.join(args.output_dir, d))
    ])
    print(f'\n总计 {len(all_embryos)} 个 embryo 组:')
    for e in all_embryos:
        memb_count = len(glob.glob(os.path.join(args.output_dir, e, 'RawMemb', '*.nii.gz')))
        nuc_count = len(glob.glob(os.path.join(args.output_dir, e, 'RawNuc', '*.nii.gz')))
        print(f'  {e}: memb={memb_count}, nuc={nuc_count}')

    print(f'\n输出目录: {args.output_dir}')


if __name__ == '__main__':
    main()
