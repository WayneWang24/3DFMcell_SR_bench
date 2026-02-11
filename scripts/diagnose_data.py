#!/usr/bin/env python
"""
诊断脚本：扫描现有数据，确认 SR→CTransformer 对接的可行性。
运行: python scripts/diagnose_data.py --sr-root <SR输出目录> --raw-root <原始数据目录>

如果不确定路径，先不带参数运行，脚本会自动搜索常见位置。
"""

import os
import sys
import glob
import argparse
from pathlib import Path

def scan_nifti(directory, max_files=5):
    """扫描目录下的 nii.gz 文件，报告 shape/dtype/range"""
    files = sorted(glob.glob(os.path.join(directory, '**', '*.nii.gz'), recursive=True))
    if not files:
        return []

    results = []
    try:
        import nibabel as nib
        import numpy as np
    except ImportError:
        return [{'path': f, 'error': 'nibabel not installed'} for f in files[:max_files]]

    for f in files[:max_files]:
        try:
            nii = nib.load(f)
            data = nii.get_fdata()
            results.append({
                'path': os.path.relpath(f, directory),
                'shape': data.shape,
                'dtype': str(nii.get_data_dtype()),
                'fdata_dtype': str(data.dtype),
                'min': float(data.min()),
                'max': float(data.max()),
                'mean': float(data.mean()),
                'voxel_size': tuple(nii.header.get_zooms()),
            })
        except Exception as e:
            results.append({'path': os.path.relpath(f, directory), 'error': str(e)})
    return results, len(files)


def check_directory_structure(root, label):
    """检查目录结构"""
    print(f"\n{'='*60}")
    print(f"  {label}: {root}")
    print(f"{'='*60}")

    if not os.path.exists(root):
        print(f"  [不存在]")
        return

    # 列出顶层目录
    for item in sorted(os.listdir(root)):
        full = os.path.join(root, item)
        if os.path.isdir(full):
            nii_count = len(glob.glob(os.path.join(full, '**', '*.nii.gz'), recursive=True))
            sub_dirs = [d for d in os.listdir(full) if os.path.isdir(os.path.join(full, d))]
            print(f"  {item}/  ({nii_count} nii.gz, 子目录: {sub_dirs[:5]})")

            # 检查 CTransformer 风格的子目录
            for sub in ['RawMemb', 'RawNuc', 'SegMemb', 'SegCell', 'SegNuc', 'AnnotatedNuc']:
                sub_path = os.path.join(full, sub)
                if os.path.exists(sub_path):
                    sub_files = glob.glob(os.path.join(sub_path, '*.nii.gz'))
                    print(f"    {sub}/  ({len(sub_files)} files)")
                    if sub_files:
                        print(f"      示例: {os.path.basename(sub_files[0])}")

        elif item.endswith('.nii.gz'):
            print(f"  {item}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sr-root', default=None, help='SR 输出根目录')
    parser.add_argument('--raw-root', default=None, help='原始数据根目录')
    parser.add_argument('--ct-data', default=None, help='CTransformer DataSource 目录')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    # ============================================================
    # 1. 搜索 SR 输出
    # ============================================================
    print("\n" + "#"*60)
    print("# 1. SR 输出数据")
    print("#"*60)

    sr_candidates = [
        args.sr_root,
        str(project_root / 'results' / 'sr_3d'),
        str(project_root / 'scripts' / 'results'),
        str(project_root / 'results'),
    ]
    sr_found = False
    for sr_dir in sr_candidates:
        if sr_dir and os.path.exists(sr_dir):
            check_directory_structure(sr_dir, "SR 输出")
            nii_files = glob.glob(os.path.join(sr_dir, '**', '*.nii.gz'), recursive=True)
            if nii_files:
                sr_found = True
                print(f"\n  [详细] 前5个文件的元数据:")
                results, total = scan_nifti(sr_dir, max_files=5)
                for r in results:
                    if 'error' in r:
                        print(f"    {r['path']}: ERROR {r['error']}")
                    else:
                        print(f"    {r['path']}:")
                        print(f"      shape={r['shape']}, dtype={r['dtype']}, "
                              f"range=[{r['min']:.2f}, {r['max']:.2f}], "
                              f"voxel={r['voxel_size']}")
                print(f"  总计 {total} 个 nii.gz 文件")

    if not sr_found:
        print("  [未找到 SR 输出] 请用 --sr-root 指定路径")

    # ============================================================
    # 2. 搜索原始数据
    # ============================================================
    print("\n" + "#"*60)
    print("# 2. 原始数据（用于 SR bench 的输入）")
    print("#"*60)

    raw_candidates = [
        args.raw_root,
        str(project_root / 'data'),
    ]
    # 也搜索 data 下的子目录
    data_dir = project_root / 'data'
    if data_dir.exists():
        for item in data_dir.iterdir():
            if item.is_dir():
                raw_candidates.append(str(item))

    raw_found = False
    for raw_dir in raw_candidates:
        if raw_dir and os.path.exists(raw_dir):
            nii_files = glob.glob(os.path.join(raw_dir, '**', '*.nii.gz'), recursive=True)
            if nii_files:
                raw_found = True
                check_directory_structure(raw_dir, "原始数据")
                print(f"\n  [详细] 前5个文件的元数据:")
                results, total = scan_nifti(raw_dir, max_files=5)
                for r in results:
                    if 'error' in r:
                        print(f"    {r['path']}: ERROR {r['error']}")
                    else:
                        print(f"    {r['path']}:")
                        print(f"      shape={r['shape']}, dtype={r['dtype']}, "
                              f"range=[{r['min']:.2f}, {r['max']:.2f}], "
                              f"voxel={r['voxel_size']}")
                print(f"  总计 {total} 个 nii.gz 文件")

    if not raw_found:
        print("  [未找到原始数据] 请用 --raw-root 指定路径")

    # ============================================================
    # 3. CTransformer DataSource
    # ============================================================
    print("\n" + "#"*60)
    print("# 3. CTransformer DataSource")
    print("#"*60)

    ct_candidates = [
        args.ct_data,
        str(project_root / 'CTransformer' / 'DataSource'),
        str(project_root / 'CTransformer' / 'data'),
    ]
    ct_found = False
    for ct_dir in ct_candidates:
        if ct_dir and os.path.exists(ct_dir):
            ct_found = True
            check_directory_structure(ct_dir, "CTransformer DataSource")

    if not ct_found:
        print("  [未找到 CTransformer DataSource]")
        print("  CTransformer 期望的目录结构:")
        print("    DataSource/RunningDataset/{embryo}/RawMemb/{embryo}_{tp}_rawMemb.nii.gz")
        print("    DataSource/RunningDataset/{embryo}/RawNuc/{embryo}_{tp}_rawNuc.nii.gz")

    # ============================================================
    # 4. CTransformer 预训练模型
    # ============================================================
    print("\n" + "#"*60)
    print("# 4. CTransformer 预训练模型")
    print("#"*60)

    ckpt_dir = project_root / 'CTransformer' / 'ckpts'
    if ckpt_dir.exists():
        pth_files = list(ckpt_dir.rglob('*.pth'))
        print(f"  找到 {len(pth_files)} 个模型文件:")
        for f in pth_files[:10]:
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"    {f.relative_to(ckpt_dir)} ({size_mb:.1f} MB)")
    else:
        print(f"  [未找到] {ckpt_dir}")
        print("  需要把 CTransformer 预训练模型放到 CTransformer/ckpts/ 下")

    # ============================================================
    # 5. CTransformer YAML 配置
    # ============================================================
    print("\n" + "#"*60)
    print("# 5. CTransformer 可用配置")
    print("#"*60)

    cfg_dir = project_root / 'CTransformer' / 'para_config'
    if cfg_dir.exists():
        yaml_files = list(cfg_dir.glob('*.yaml'))
        print(f"  找到 {len(yaml_files)} 个配置文件:")
        for f in yaml_files:
            print(f"    {f.name}")
    else:
        print(f"  [未找到] {cfg_dir}")

    # ============================================================
    # 6. GT 分割标注（用于评估）
    # ============================================================
    print("\n" + "#"*60)
    print("# 6. Ground Truth 分割标注")
    print("#"*60)

    gt_patterns = ['**/SegCell/**/*.nii.gz', '**/SegMemb/**/*.nii.gz',
                   '**/SegNuc/**/*.nii.gz', '**/AnnotatedNuc/**/*.nii.gz']
    gt_found = False
    search_roots = [str(project_root / 'data'), str(project_root / 'CTransformer')]
    if args.raw_root:
        search_roots.append(args.raw_root)
    if args.ct_data:
        search_roots.append(args.ct_data)

    for search_root in search_roots:
        if not os.path.exists(search_root):
            continue
        for pattern in gt_patterns:
            files = glob.glob(os.path.join(search_root, pattern), recursive=True)
            if files:
                gt_found = True
                # 按类型分组
                seg_type = pattern.split('/')[1]
                print(f"  {seg_type} in {search_root}: {len(files)} files")
                if files:
                    print(f"    示例: {os.path.basename(files[0])}")

    if not gt_found:
        print("  [未找到 GT 标注] 如果有 GT SegCell，可以做定量评估 (Dice/IoU)")
        print("  如果没有，只能做定性比较（细胞数量、形态等）")

    # ============================================================
    # 7. 时间序列检查（谱系追踪需要）
    # ============================================================
    print("\n" + "#"*60)
    print("# 7. 时间序列数据（谱系追踪需要）")
    print("#"*60)

    # 搜索多时间点数据
    found_timeseries = False
    for search_root in search_roots:
        if not os.path.exists(search_root):
            continue
        memb_files = glob.glob(os.path.join(search_root, '**', '*rawMemb.nii.gz'), recursive=True)
        if len(memb_files) > 1:
            found_timeseries = True
            # 按 embryo 分组
            embryos = {}
            for f in memb_files:
                name = os.path.basename(f)
                parts = name.split('_')
                if len(parts) >= 2:
                    embryo = parts[0]
                    embryos.setdefault(embryo, []).append(f)
            for embryo, files in embryos.items():
                print(f"  胚胎 '{embryo}': {len(files)} 个时间点")
                print(f"    路径: {os.path.dirname(files[0])}")

    if not found_timeseries:
        print("  [未找到多时间点数据]")
        print("  谱系追踪需要同一胚胎的多个时间点 (至少 10+)")
        print("  如果只有单时间点，只能做分割对比，不能做谱系对比")

    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "#"*60)
    print("# 总结")
    print("#"*60)
    print(f"  SR 输出:        {'有' if sr_found else '无'}")
    print(f"  原始数据:       {'有' if raw_found else '无'}")
    print(f"  CTransformer:   {'有' if ct_found else '无'}")
    print(f"  GT 标注:        {'有' if gt_found else '无'}")
    print(f"  时间序列:       {'有' if found_timeseries else '无'}")
    print()
    print("  请将此输出反馈给我，我会据此生成完整的对比 pipeline。")


if __name__ == '__main__':
    main()
