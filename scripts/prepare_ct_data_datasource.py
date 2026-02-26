#!/usr/bin/env python
"""
将 SR 增强后的 DataSource 数据转换为 CTransformer 期望的目录结构，并预 resize。

DataSource SR 流程:
  DataSource/{embryo}/RawMemb/ → SR → fix_intensity → sr_fixed/{embryo}/*.nii.gz

本脚本:
  sr_fixed/{embryo}/*.nii.gz  +  DataSource/{embryo}/RawNuc/*.nii.gz
  → ct_input/{embryo}/RawMemb/{embryo}_{tp}_rawMemb.nii.gz  (resized)
  → ct_input/{embryo}/RawNuc/{embryo}_{tp}_rawNuc.nii.gz    (resized)

与 prepare_ct_data.py 的区别:
  - 输入文件名: {embryo}_{tp}_rawMemb_sr_x4.nii.gz (DataSource 格式)
  - RawNuc 来源: DataSource/{embryo}/RawNuc/ (独立目录)
  - 胚胎自动发现: 从 sr-dir 子目录

使用方法:
  python scripts/prepare_ct_data_datasource.py \
    --sr-dir scripts/results/exp2_edsr/sr_fixed \
    --datasource /home/vm/workspace/DataSource/RunningDataset \
    --output-dir scripts/results/exp2_edsr/ct_input \
    --target-size 256 384 224
"""

import os
import sys
import glob
import argparse

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


TARGET_SIZE = (256, 384, 224)


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
        out_nii = nib.Nifti1Image(data, affine)
    else:
        resized = zoom(data, factors, order=1)
        for i in range(3):
            affine[i, i] = affine[i, i] / factors[i]
        out_nii = nib.Nifti1Image(resized, affine)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(out_nii, output_path)


def parse_tp_from_filename(filename, embryo_name):
    """
    从 DataSource 格式文件名解析时间点。

    支持:
      {embryo}_{tp}_rawMemb.nii.gz
      {embryo}_{tp}_rawMemb_sr_x4.nii.gz
      {embryo}_{tp}_rawNuc.nii.gz

    返回: tp 字符串 (如 '050')
    """
    stem = filename.replace('.nii.gz', '')
    # 去掉 SR 后缀
    for suffix in ['_sr_x2', '_sr_x4', '_sr_x8']:
        stem = stem.replace(suffix, '')

    # 去掉 embryo 前缀
    prefix = embryo_name + '_'
    if stem.startswith(prefix):
        remainder = stem[len(prefix):]
        tp = remainder.split('_')[0]
        return tp

    # Fallback: split('_')[1]
    parts = stem.split('_')
    if len(parts) >= 2:
        return parts[1]

    return None


def prepare_embryo(sr_embryo_dir, nuc_dir, embryo_name, output_root, target_size):
    """处理单个胚胎: SR memb + 原始 nuc → CTransformer 输入"""
    sr_files = sorted(glob.glob(os.path.join(sr_embryo_dir, '*.nii.gz')))
    if not sr_files:
        return 0

    memb_out = os.path.join(output_root, embryo_name, 'RawMemb')
    nuc_out = os.path.join(output_root, embryo_name, 'RawNuc')
    os.makedirs(memb_out, exist_ok=True)
    os.makedirs(nuc_out, exist_ok=True)

    count = 0
    missing_nuc = 0
    for sr_path in sr_files:
        filename = os.path.basename(sr_path)
        tp = parse_tp_from_filename(filename, embryo_name)
        if tp is None:
            print(f'    [跳过] 无法解析时间点: {filename}')
            continue

        target_memb = os.path.join(memb_out, f'{embryo_name}_{tp}_rawMemb.nii.gz')
        target_nuc = os.path.join(nuc_out, f'{embryo_name}_{tp}_rawNuc.nii.gz')

        # Memb: resize SR 输出
        resize_and_save(sr_path, target_memb, target_size)

        # Nuc: resize 原始 nuc
        nuc_path = os.path.join(nuc_dir, f'{embryo_name}_{tp}_rawNuc.nii.gz')
        if os.path.exists(nuc_path):
            resize_and_save(nuc_path, target_nuc, target_size)
        else:
            missing_nuc += 1

        count += 1
        if count % 20 == 0 or count == 1:
            print(f'    {embryo_name}: {count}/{len(sr_files)}')

    if missing_nuc > 0:
        print(f'    [警告] {embryo_name}: {missing_nuc} 个时间点缺少 RawNuc')

    return count


def main():
    parser = argparse.ArgumentParser(description='准备 DataSource SR 数据的 CTransformer 输入')
    parser.add_argument('--sr-dir', required=True,
                        help='SR 修复后输出目录 (含 {embryo}/ 子目录)')
    parser.add_argument('--datasource', required=True,
                        help='DataSource/RunningDataset 路径')
    parser.add_argument('--output-dir', required=True,
                        help='CTransformer 输入输出路径')
    parser.add_argument('--target-size', nargs=3, type=int,
                        default=[256, 384, 224],
                        help='目标体积尺寸 (X Y Z), 默认 256 384 224')
    parser.add_argument('--embryos', nargs='*', default=None,
                        help='指定胚胎列表 (不指定则自动发现)')
    args = parser.parse_args()

    target_size = tuple(args.target_size)
    print(f'目标尺寸: {target_size}')

    # 确定胚胎列表
    if args.embryos:
        embryos = args.embryos
    else:
        embryos = sorted([
            d for d in os.listdir(args.sr_dir)
            if os.path.isdir(os.path.join(args.sr_dir, d))
        ])

    if not embryos:
        print(f'[错误] SR 目录下无胚胎子目录: {args.sr_dir}')
        sys.exit(1)

    print(f'发现 {len(embryos)} 个胚胎')

    total = 0
    skipped = 0
    for embryo in embryos:
        sr_embryo_dir = os.path.join(args.sr_dir, embryo)
        nuc_dir = os.path.join(args.datasource, embryo, 'RawNuc')

        if not os.path.isdir(sr_embryo_dir):
            print(f'  [跳过] SR 目录不存在: {sr_embryo_dir}')
            skipped += 1
            continue

        # 检查是否已处理
        existing = glob.glob(os.path.join(args.output_dir, embryo, 'RawMemb', '*.nii.gz'))
        sr_count = len(glob.glob(os.path.join(sr_embryo_dir, '*.nii.gz')))
        if len(existing) >= sr_count > 0:
            print(f'  [跳过] {embryo}: 已有 {len(existing)} 个 >= SR {sr_count} 个')
            skipped += 1
            continue

        print(f'  [处理] {embryo}')
        if not os.path.exists(nuc_dir):
            print(f'    [警告] RawNuc 目录不存在: {nuc_dir}')

        n = prepare_embryo(sr_embryo_dir, nuc_dir, embryo, args.output_dir, target_size)
        print(f'  {embryo}: {n} 个时间点')
        total += n

    # 汇总
    all_embryos = sorted([
        d for d in os.listdir(args.output_dir)
        if os.path.isdir(os.path.join(args.output_dir, d))
    ]) if os.path.exists(args.output_dir) else []

    print(f'\n===== 汇总 =====')
    print(f'处理: {len(embryos) - skipped} 个胚胎, {total} 个新时间点')
    print(f'跳过: {skipped} 个胚胎')
    print(f'输出目录: {args.output_dir} ({len(all_embryos)} 个胚胎)')
    for e in all_embryos:
        mc = len(glob.glob(os.path.join(args.output_dir, e, 'RawMemb', '*.nii.gz')))
        nc = len(glob.glob(os.path.join(args.output_dir, e, 'RawNuc', '*.nii.gz')))
        print(f'  {e}: memb={mc}, nuc={nc}')

    print(f'\n所有体积已预 resize 到 {target_size}')


if __name__ == '__main__':
    main()
