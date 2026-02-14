#!/usr/bin/env python
"""
SR 输出后处理: 修复膜结构亮度/对比度。

将 SR 输出的强度分布匹配到原始数据，保留 SR 的空间细节，恢复膜对比度。

用法:
  python scripts/fix_sr_intensity.py \
    --sr-dir scripts/results/eval_outputs/sr_3d_fastz_xz \
    --raw-dir data/raw_cell_datasets/fastz_volumes \
    --output-dir scripts/results/eval_outputs/sr_3d_fastz_xz_fixed

  也可直接覆盖原文件 (--inplace):
  python scripts/fix_sr_intensity.py \
    --sr-dir scripts/results/eval_outputs/sr_3d_fastz_xz \
    --raw-dir data/raw_cell_datasets/fastz_volumes \
    --inplace
"""

import os
import glob
import argparse
import numpy as np
import nibabel as nib


def histogram_match(source, reference):
    """
    将 source 的直方图匹配到 reference。
    保留 source 的空间结构，替换强度分布。
    """
    src_flat = source.ravel()
    ref_flat = reference.ravel()

    # 计算排序索引
    src_sort_idx = np.argsort(src_flat)
    ref_sort_idx = np.argsort(ref_flat)

    # 用 reference 的排序值替换 source 的排序值
    # 如果两者长度不同，需要插值
    src_n = len(src_flat)
    ref_n = len(ref_flat)

    ref_sorted = ref_flat[ref_sort_idx]

    if src_n == ref_n:
        matched_flat = np.empty_like(src_flat)
        matched_flat[src_sort_idx] = ref_sorted
    else:
        # 插值: 把 reference 的分位数映射到 source 的每个像素
        ref_quantiles = np.linspace(0, 1, ref_n)
        src_quantiles = np.linspace(0, 1, src_n)
        ref_interp = np.interp(src_quantiles, ref_quantiles, ref_sorted)
        matched_flat = np.empty_like(src_flat)
        matched_flat[src_sort_idx] = ref_interp

    return matched_flat.reshape(source.shape)


def percentile_match(source, reference):
    """
    简单线性匹配: 把 source 的 [p1, p99] 映射到 reference 的 [p1, p99]。
    比直方图匹配更快，适合大体积数据。
    """
    src_p1, src_p99 = np.percentile(source, [1, 99])
    ref_p1, ref_p99 = np.percentile(reference, [1, 99])

    if src_p99 - src_p1 < 1e-6:
        return source

    # 线性映射
    matched = (source - src_p1) / (src_p99 - src_p1) * (ref_p99 - ref_p1) + ref_p1
    matched = np.clip(matched, 0, max(ref_p99 * 1.5, source.max()))
    return matched


def find_matching_raw(sr_filename, raw_dir):
    """根据 SR 文件名找到对应的原始文件"""
    # SR 文件名: memb_Series001_sr_x4.nii.gz 或 memb_L_1_sr_x4.nii.gz
    basename = os.path.basename(sr_filename)
    # 去掉 SR 后缀
    raw_name = basename
    for suffix in ['_sr_x2', '_sr_x4', '_sr_x8']:
        raw_name = raw_name.replace(suffix, '')

    raw_path = os.path.join(raw_dir, raw_name)
    if os.path.exists(raw_path):
        return raw_path
    return None


def process_one_model(model_dir, raw_dir, output_dir, method, inplace):
    """处理一个 SR 模型的所有输出"""
    sr_files = sorted(glob.glob(os.path.join(model_dir, '*.nii.gz')))
    if not sr_files:
        return 0

    count = 0
    for sr_path in sr_files:
        raw_path = find_matching_raw(sr_path, raw_dir)
        if raw_path is None:
            print(f'  [跳过] 找不到原始文件: {os.path.basename(sr_path)}')
            continue

        # 加载
        sr_nii = nib.load(sr_path)
        sr_data = sr_nii.get_fdata(dtype=np.float32)
        raw_data = nib.load(raw_path).get_fdata(dtype=np.float32)

        # 打印修复前信息
        sr_p1, sr_p99 = np.percentile(sr_data, [1, 99])
        raw_p1, raw_p99 = np.percentile(raw_data, [1, 99])

        # 匹配
        if method == 'histogram':
            # 对大体积做采样加速
            if sr_data.size > 10_000_000:
                # 用下采样版本计算映射，然后应用到全量
                fixed = percentile_match(sr_data, raw_data)
            else:
                fixed = histogram_match(sr_data, raw_data)
        else:
            fixed = percentile_match(sr_data, raw_data)

        fixed_p1, fixed_p99 = np.percentile(fixed, [1, 99])

        # 保存
        if inplace:
            save_path = sr_path
        else:
            rel = os.path.relpath(sr_path, os.path.dirname(model_dir))
            save_path = os.path.join(output_dir, rel)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        out_nii = nib.Nifti1Image(fixed.astype(sr_nii.get_data_dtype()), sr_nii.affine)
        nib.save(out_nii, save_path)

        count += 1
        if count <= 3 or count % 10 == 0:
            print(f'  {os.path.basename(sr_path)}:')
            print(f'    SR  p1={sr_p1:.1f}, p99={sr_p99:.1f}')
            print(f'    Raw p1={raw_p1:.1f}, p99={raw_p99:.1f}')
            print(f'    Fix p1={fixed_p1:.1f}, p99={fixed_p99:.1f}')

    return count


def main():
    parser = argparse.ArgumentParser(description='修复 SR 输出的强度分布')
    parser.add_argument('--sr-dir', required=True,
                        help='SR 输出目录 (含各模型子目录)')
    parser.add_argument('--raw-dir', required=True,
                        help='原始 NIfTI 数据目录 (memb_*.nii.gz)')
    parser.add_argument('--output-dir', default=None,
                        help='输出目录 (默认在 sr-dir 旁边加 _fixed 后缀)')
    parser.add_argument('--method', choices=['percentile', 'histogram'],
                        default='percentile',
                        help='匹配方法: percentile (快) 或 histogram (精确)')
    parser.add_argument('--inplace', action='store_true',
                        help='直接覆盖原文件')
    args = parser.parse_args()

    if not args.inplace and args.output_dir is None:
        args.output_dir = args.sr_dir.rstrip('/') + '_fixed'

    print(f'SR 目录:   {args.sr_dir}')
    print(f'原始目录:  {args.raw_dir}')
    print(f'输出:      {"覆盖原文件" if args.inplace else args.output_dir}')
    print(f'方法:      {args.method}')
    print()

    # 检查是否有子目录 (各 SR 模型)
    subdirs = sorted([
        d for d in os.listdir(args.sr_dir)
        if os.path.isdir(os.path.join(args.sr_dir, d))
    ])

    if subdirs:
        # 有子目录: 逐模型处理
        for model_name in subdirs:
            model_path = os.path.join(args.sr_dir, model_name)
            print(f'[{model_name}]')
            n = process_one_model(model_path, args.raw_dir,
                                  args.output_dir, args.method, args.inplace)
            print(f'  处理 {n} 个文件\n')
    else:
        # 无子目录: 直接处理
        n = process_one_model(args.sr_dir, args.raw_dir,
                              args.output_dir, args.method, args.inplace)
        print(f'处理 {n} 个文件')

    print('完成!')


if __name__ == '__main__':
    main()
