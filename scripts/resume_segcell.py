#!/usr/bin/env python
"""
恢复脚本：只跑 SegCell (membrane→cell watershed)，跳过已完成的，带超时。

用法:
  cd /home/vm/workspace/3DFMcell_SR_bench
  python scripts/resume_segcell.py \
    --seg-root scripts/results/seg_comparison/singleS_xz \
    --embryos singleS-original singleS-swinir-x4-singleS-xz-p1p99 \
    --timeout 600 \
    --workers 8
"""

import os
import sys
import glob
import argparse
import multiprocessing as mp
from multiprocessing import TimeoutError as MPTimeoutError
from tqdm import tqdm

# 添加 CTransformer 到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'CTransformer'))

from segmentation_utils.ProcessLib import instance_segmentation_watershed


class FakeArgs(dict):
    """模拟 CTransformer 的 integrated_args"""
    def __getattr__(self, name):
        return self.get(name)


def run_single_with_timeout(pool, func, params, timeout):
    """用 apply_async + timeout 跑单个任务"""
    result = pool.apply_async(func, (params,))
    try:
        result.get(timeout=timeout)
        return True
    except MPTimeoutError:
        return False
    except Exception as e:
        print(f'  [错误] {e}')
        return False


def main():
    parser = argparse.ArgumentParser(description='恢复 SegCell (跳过已完成，带超时)')
    parser.add_argument('--seg-root', required=True,
                        help='分割输出根目录 (如 seg_comparison/singleS_xz)')
    parser.add_argument('--embryos', nargs='+', required=True,
                        help='要处理的 embryo 名称列表')
    parser.add_argument('--timeout', type=int, default=600,
                        help='每个时间点超时秒数 (默认 600=10分钟)')
    parser.add_argument('--workers', type=int, default=8,
                        help='并行 worker 数 (默认 8)')
    args = parser.parse_args()

    integrated_args = FakeArgs({
        'is_nuc_labelled': False,
        'is_nuc_predicted': False,
        'is_nuc_predicted_and_localmin': False,
        'is_4D': False,
        'mem_edt_threshold': 9,
        'topology_constraint': False,
        'output_data_path': args.seg_root,
    })

    for embryo in args.embryos:
        seg_memb_dir = os.path.join(args.seg_root, embryo, 'SegMemb')
        seg_cell_dir = os.path.join(args.seg_root, embryo, 'SegCell')
        os.makedirs(seg_cell_dir, exist_ok=True)

        memb_files = sorted(glob.glob(os.path.join(seg_memb_dir, '*segMemb.nii.gz')))
        if not memb_files:
            print(f'[跳过] {embryo}: 没有 SegMemb 文件')
            continue

        # 过滤掉已有 SegCell 的
        todo = []
        for f in memb_files:
            base = os.path.basename(f)
            parts = base.split("_")[:2]
            cell_name = "_".join(parts) + "_segCell.nii.gz"
            cell_path = os.path.join(seg_cell_dir, cell_name)
            if os.path.exists(cell_path):
                continue
            todo.append(f)

        print(f'\n[{embryo}] SegMemb={len(memb_files)}, 已完成={len(memb_files)-len(todo)}, 待处理={len(todo)}')

        if not todo:
            print(f'  全部已完成，跳过')
            continue

        # 逐个跑，带超时
        pool = mp.Pool(args.workers)
        done = 0
        skipped = 0
        for f in tqdm(todo, desc=f'{embryo} segCell'):
            tp_name = "_".join(os.path.basename(f).split("_")[:2])
            params = [seg_cell_dir, f, None, integrated_args]
            ok = run_single_with_timeout(pool, instance_segmentation_watershed, params, args.timeout)
            if ok:
                done += 1
            else:
                skipped += 1
                print(f'  [超时/失败] {tp_name} (>{args.timeout}s)，跳过')

        pool.terminate()
        pool.join()
        print(f'[{embryo}] 完成={done}, 超时跳过={skipped}')

    print('\n全部处理完毕!')


if __name__ == '__main__':
    main()
