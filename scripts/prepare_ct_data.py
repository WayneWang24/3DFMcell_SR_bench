#!/usr/bin/env python
"""
将原始数据和 SR 输出转换为 CTransformer 期望的目录结构，并预 resize。

CTransformer 期望:
  {output_dir}/{embryo}/RawMemb/{embryo}_{tp}_rawMemb.nii.gz
  {output_dir}/{embryo}/RawNuc/{embryo}_{tp}_rawNuc.nii.gz

本脚本:
1. 原始数据: 加载 memb + nuc, resize 到目标尺寸, 保存
2. SR 数据: 加载 SR memb + 原始 nuc, 都 resize 到目标尺寸, 保存

预 resize 的好处:
- SR 质量优势在 resize 后仍保留 (类似 4K→1080p 降采样)
- CTransformer 加载时无需再 resize, 速度提升 10-50x

使用方法:
  python scripts/prepare_ct_data.py \
    --raw-dir data/raw_cell_datasets/fastz_volumes \
    --sr-dir scripts/results/eval_outputs/sr_3d_fastz_xz \
    --output-dir data/ct_input \
    --condition fastz \
    --target-size 256 384 224
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


TARGET_SIZE = (256, 384, 224)  # CTransformer 默认输入尺寸


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


def resize_and_save(input_path, output_path, target_size):
    """加载 NIfTI, resize 到目标尺寸, 保存为 float32"""
    if os.path.exists(output_path):
        return

    nii = nib.load(input_path)
    data = nii.get_fdata(dtype=np.float32)
    affine = nii.affine.copy()

    orig_shape = data.shape
    factors = tuple(t / o for t, o in zip(target_size, orig_shape))

    if all(abs(f - 1.0) < 1e-6 for f in factors):
        # 尺寸已一致
        out_nii = nib.Nifti1Image(data, affine)
    else:
        resized = zoom(data, factors, order=1)  # 线性插值
        # 更新 affine
        for i in range(3):
            affine[i, i] = affine[i, i] / factors[i]
        out_nii = nib.Nifti1Image(resized, affine)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(out_nii, output_path)


def prepare_original(raw_dir, embryo_name, output_root, target_size):
    """原始数据: 加载 memb + nuc, resize, 保存"""
    glob_pattern = os.path.join(raw_dir, 'memb_*.nii.gz')
    memb_files = sorted(glob.glob(glob_pattern))

    print(f'  [DEBUG] raw_dir = {raw_dir}')
    print(f'  [DEBUG] raw_dir exists = {os.path.exists(raw_dir)}')
    print(f'  [DEBUG] glob pattern = {glob_pattern}')
    print(f'  [DEBUG] matched files = {len(memb_files)}')
    if memb_files:
        print(f'  [DEBUG] first file = {memb_files[0]}')
        print(f'  [DEBUG] last file  = {memb_files[-1]}')
    elif os.path.exists(raw_dir):
        # 列出目录内容帮助调试
        all_files = os.listdir(raw_dir)
        nii_files = [f for f in all_files if f.endswith('.nii.gz')]
        print(f'  [DEBUG] 目录内 .nii.gz 文件数 = {len(nii_files)}')
        if nii_files:
            print(f'  [DEBUG] 前5个: {nii_files[:5]}')

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

        resize_and_save(memb_path, target_memb, target_size)

        nuc_path = find_nuc_file(raw_dir, filename)
        if nuc_path:
            resize_and_save(nuc_path, target_nuc, target_size)

        count += 1
        if count % 5 == 0 or count == 1:
            print(f'  {embryo_name}: {count} done')
    return count


def prepare_sr(sr_model_dir, raw_dir, embryo_name, output_root, target_size):
    """SR 数据: SR memb + 原始 nuc, 都 resize 到目标尺寸"""
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

        # memb: resize SR 输出
        resize_and_save(sr_path, target_memb, target_size)

        # nuc: resize 原始 nuc (直接从原始 resize 到目标, 不需要先插值到 SR Z 尺寸)
        nuc_path = find_nuc_file(raw_dir, filename)
        if nuc_path:
            resize_and_save(nuc_path, target_nuc, target_size)

        count += 1
        if count % 5 == 0 or count == 1:
            print(f'  {embryo_name}: {count} done')
    return count


def main():
    parser = argparse.ArgumentParser(description='准备 CTransformer 输入数据 (含预 resize)')
    parser.add_argument('--raw-dir', required=True,
                        help='原始数据目录 (包含 memb_*.nii.gz 和 nuc_*.nii.gz)')
    parser.add_argument('--sr-dir', required=True,
                        help='SR 输出目录 (包含各模型子目录)')
    parser.add_argument('--output-dir', required=True,
                        help='输出路径')
    parser.add_argument('--condition', required=True,
                        help='条件名称 (fastz / singleS)')
    parser.add_argument('--target-size', nargs=3, type=int,
                        default=[256, 384, 224],
                        help='目标体积尺寸 (X Y Z), 默认 256 384 224')
    args = parser.parse_args()

    target_size = tuple(args.target_size)
    print(f'目标尺寸: {target_size}')

    # ============================================================
    # 1. 原始数据
    # ============================================================
    # CTransformer 用 split('_')[:2] 解析文件名，embryo 名不能含下划线
    embryo_orig = f'{args.condition}-original'
    print(f'\n[原始数据] {embryo_orig}')
    n = prepare_original(args.raw_dir, embryo_orig, args.output_dir, target_size)
    print(f'[原始数据] {embryo_orig}: {n} 个时间点')

    # ============================================================
    # 2. 各 SR 模型
    # ============================================================
    if not os.path.exists(args.sr_dir):
        print(f"[警告] SR 目录不存在: {args.sr_dir}")
        print(f"  跳过 SR 模型处理，只处理原始数据")
    else:
        model_dirs = sorted([
            d for d in os.listdir(args.sr_dir)
            if os.path.isdir(os.path.join(args.sr_dir, d))
        ])

        for model_name in model_dirs:
            model_path = os.path.join(args.sr_dir, model_name)
            embryo_sr = f'{args.condition}-{model_name}'.replace('_', '-')
            print(f'\n[SR 模型] {embryo_sr}')
            n = prepare_sr(model_path, args.raw_dir, embryo_sr, args.output_dir, target_size)
            if n > 0:
                print(f'[SR 模型] {embryo_sr}: {n} 个时间点')

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
    print(f'所有体积已预 resize 到 {target_size}, CTransformer 无需再 resize')


if __name__ == '__main__':
    main()
