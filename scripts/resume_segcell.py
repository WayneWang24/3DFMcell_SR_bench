#!/usr/bin/env python
"""
恢复脚本：只跑 SegCell (membrane→cell watershed)，跳过已完成的，带超时。

用法:
  cd /home/vm/workspace/3DFMcell_SR_bench/CTransformer
  python ../scripts/resume_segcell.py \
    --seg-root ../scripts/results/seg_comparison/singleS_xz \
    --embryos singleS-original singleS-swinir-x4-singleS-xz-p1p99 \
    --timeout 600 \
    --workers 8
"""

import os
import sys
import glob
import argparse
import traceback
import multiprocessing as mp
from tqdm import tqdm

# 添加 CTransformer 到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CT_ROOT = os.path.join(PROJECT_ROOT, 'CTransformer')
sys.path.insert(0, CT_ROOT)

from segmentation_utils.ProcessLib import instance_segmentation_watershed


class FakeArgs(dict):
    """模拟 CTransformer 的 integrated_args"""
    def __getattr__(self, name):
        return self.get(name)


def worker_wrapper(params):
    """在 worker 中跑并捕获完整 traceback"""
    try:
        instance_segmentation_watershed(params)
        return (True, None)
    except Exception:
        return (False, traceback.format_exc())


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

    # 使用绝对路径
    seg_root = os.path.abspath(args.seg_root)

    integrated_args = FakeArgs({
        'is_nuc_labelled': False,
        'is_nuc_predicted': False,
        'is_nuc_predicted_and_localmin': False,
        'is_4D': False,
        'mem_edt_threshold': 9,
        'topology_constraint': False,
        'output_data_path': seg_root,
    })

    # 收集所有待处理任务
    all_tasks = []
    for embryo in args.embryos:
        seg_memb_dir = os.path.join(seg_root, embryo, 'SegMemb')
        seg_cell_dir = os.path.join(seg_root, embryo, 'SegCell')
        os.makedirs(seg_cell_dir, exist_ok=True)

        memb_files = sorted(glob.glob(os.path.join(seg_memb_dir, '*segMemb.nii.gz')))
        if not memb_files:
            print(f'[跳过] {embryo}: 没有 SegMemb 文件')
            continue

        todo = []
        for f in memb_files:
            base = os.path.basename(f)
            parts = base.split("_")[:2]
            cell_name = "_".join(parts) + "_segCell.nii.gz"
            cell_path = os.path.join(seg_cell_dir, cell_name)
            if os.path.exists(cell_path):
                continue
            todo.append(f)

        print(f'[{embryo}] SegMemb={len(memb_files)}, 已完成={len(memb_files)-len(todo)}, 待处理={len(todo)}')

        for f in todo:
            all_tasks.append((embryo, f, seg_cell_dir))

    if not all_tasks:
        print('没有待处理任务')
        return

    print(f'\n总计 {len(all_tasks)} 个任务, workers={args.workers}, timeout={args.timeout}s')

    # 用一个 Pool 处理所有任务，逐个提交带超时
    pool = mp.Pool(args.workers)
    done = 0
    timeout_count = 0
    error_count = 0

    for embryo, f, seg_cell_dir in tqdm(all_tasks, desc='segCell'):
        tp_name = "_".join(os.path.basename(f).split("_")[:2])
        params = [seg_cell_dir, f, None, integrated_args]

        result = pool.apply_async(worker_wrapper, (params,))
        try:
            success, tb = result.get(timeout=args.timeout)
            if success:
                done += 1
            else:
                error_count += 1
                print(f'\n  [错误] {tp_name}:')
                print(tb)
        except mp.TimeoutError:
            timeout_count += 1
            print(f'\n  [超时] {tp_name} (>{args.timeout}s)，跳过')
            # 超时后需要重建 pool，因为 worker 还在跑
            pool.terminate()
            pool.join()
            pool = mp.Pool(args.workers)

    pool.close()
    pool.join()

    print(f'\n完成={done}, 超时={timeout_count}, 错误={error_count}')


if __name__ == '__main__':
    main()
