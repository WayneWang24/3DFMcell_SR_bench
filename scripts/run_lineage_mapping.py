#!/usr/bin/env python
"""
Step 6 Wrapper: 谱系映射 — 将 AnnotatedNuc 映射到 SegCell 结果。

原始 CTransformer/6_build_cell_shape_map.py 使用硬编码路径，
本脚本提供命令行接口，方便在不同数据集上运行。

用法:
  python scripts/run_lineage_mapping.py \
    --seg-root /path/to/seg_output \
    --annotated-root /path/to/DataSource/RunningDataset \
    --name-dict /path/to/name_dictionary.csv \
    --cell-fate /path/to/CellFate.xls \
    --output /path/to/lineage_output \
    --embryos 170704plc1p1 200113plc1p2
"""

import os
import sys
import argparse
import multiprocessing as mp
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

# 添加 CTransformer 到 path
CT_ROOT = os.path.join(os.path.dirname(__file__), '..', 'CTransformer')
sys.path.insert(0, os.path.abspath(CT_ROOT))

from lineage_gui_utils.lineage_tree import construct_celltree
from utils.data_io import nib_load, nib_save, check_folder


def combine_division_mp(para):
    """合并分裂中的细胞 (从 6_build_cell_shape_map.py 移植)"""
    import skimage.morphology
    from scipy import ndimage
    from scipy.ndimage.morphology import binary_closing
    from segmentation_utils.ProcessLib import line_weight_integral

    segmented_cell_file_path = para[0]
    predicted_memb_file_path = para[1]
    annotated_nuc_file_path = para[2]
    cell_tree = para[3]
    name_label_dict = para[4]
    label_name_dict = para[5]

    this_tp = os.path.basename(predicted_memb_file_path).split('_')[1]

    pred_memb_map = nib_load(predicted_memb_file_path)
    seg_bin = (pred_memb_map > 0.93 * pred_memb_map.max()).astype(float)

    annotated_nuc = nib_load(annotated_nuc_file_path)
    nucleus_marker_footprint = skimage.morphology.ball(7 - int(int(this_tp) / 100))
    annotated_nuc = ndimage.grey_erosion(annotated_nuc, footprint=nucleus_marker_footprint)
    seg_cell = nib_load(segmented_cell_file_path)

    labels = np.unique(annotated_nuc)[1:].tolist()
    processed_labels = []
    output_seg_cell = seg_cell.copy()
    cell_labels = np.unique(seg_cell).tolist()

    for one_label in labels:
        try:
            one_times = cell_tree[label_name_dict[one_label]].data.get_time()
        except (KeyError, AttributeError):
            continue
        if any(time < int(this_tp) for time in one_times):
            continue
        if one_label in processed_labels:
            continue
        parent_label = cell_tree.parent(label_name_dict[one_label])
        if parent_label is None:
            continue
        try:
            another_label = [name_label_dict[a.tag] for a in cell_tree.children(parent_label.tag)]
            another_label.remove(one_label)
            another_label = another_label[0]
        except (KeyError, ValueError, IndexError):
            continue

        if (one_label not in cell_labels) or (another_label not in cell_labels):
            continue

        x0 = np.stack(np.where(annotated_nuc == one_label)).squeeze().tolist()
        x1 = np.stack(np.where(annotated_nuc == another_label)).squeeze().tolist()
        edge_weight = line_weight_integral(x0=x0, x1=x1, weight_volume=seg_bin)
        if edge_weight == 0:
            mask = np.logical_or(seg_cell == one_label, seg_cell == another_label)
            mask = binary_closing(mask, structure=np.ones((3, 3, 3)))
            output_seg_cell[mask] = name_label_dict[parent_label.tag]
            one_times.remove(int(this_tp))
            another_times = cell_tree[label_name_dict[another_label]].data.get_time()
            another_times.remove(int(this_tp))
            cell_tree[label_name_dict[one_label]].data.set_time(one_times)
            cell_tree[label_name_dict[another_label]].data.set_time(another_times)
        processed_labels += [one_label, another_label]

    nib_save(output_seg_cell, segmented_cell_file_path)


def running_reassign_cellular_map(para):
    """将 AnnotatedNuc 标号映射到 SegCell (从 6_build_cell_shape_map.py 移植)"""
    import skimage.morphology
    from scipy import ndimage

    segCellniigzpath = para[0]
    cell_tree_embryo = para[1]
    annotated_niigz_path = para[2]
    embryo_name = para[3]
    cell2fate = para[4]
    label_name_dict = para[5]
    output_saving_path = para[6]

    tp_this_str = os.path.basename(segCellniigzpath).split('_')[1]
    segmented_arr = nib_load(segCellniigzpath).astype(int)
    annotated_nuc_arr = nib_load(annotated_niigz_path).astype(int)
    nucleus_marker_footprint = skimage.morphology.ball(7 - int(int(tp_this_str) / 100))
    annotated_nuc_arr = ndimage.grey_erosion(annotated_nuc_arr, footprint=nucleus_marker_footprint)

    new_cellular_arr = np.zeros(segmented_arr.shape)
    cells_list_this = np.unique(annotated_nuc_arr)[1:]
    mapping_cellular_dict = {}
    dividing_cells = []
    lossing_cells = []
    late_processing_cells = []

    for cell_index in cells_list_this:
        this_cell_fate = cell2fate.get(label_name_dict.get(cell_index, ''), 'Unspecified')
        if this_cell_fate == 'Death':
            late_processing_cells.append(cell_index)
        else:
            locations_cellular = np.where(annotated_nuc_arr == cell_index)
            centre_len_tmp = 0
            x_tmp = locations_cellular[0][centre_len_tmp]
            y_tmp = locations_cellular[1][centre_len_tmp]
            z_tmp = locations_cellular[2][centre_len_tmp]
            segmented_arr_index = int(segmented_arr[x_tmp, y_tmp, z_tmp])

            if segmented_arr_index in mapping_cellular_dict.keys() or segmented_arr_index == 0:
                is_found = False
                assigned_cell_name = label_name_dict.get(
                    mapping_cellular_dict.get(segmented_arr_index, 'ZERO'), 'ZERO')
                this_cell_name = label_name_dict.get(cell_index, '')

                IS_ZERO_BACK = (assigned_cell_name == 'ZERO' or segmented_arr_index == 0)

                if not IS_ZERO_BACK:
                    parent_node_occupied = cell_tree_embryo.parent(assigned_cell_name)
                    parent_node_this = cell_tree_embryo.parent(this_cell_name)

                if not IS_ZERO_BACK and parent_node_this is not None and parent_node_occupied is not None \
                        and parent_node_this.tag == parent_node_occupied.tag:
                    cell_lifecycle = cell_tree_embryo[this_cell_name].data.get_time()
                    nuc_dividing_reasonable = len(cell_lifecycle) // 4
                    if int(tp_this_str) < cell_lifecycle[nuc_dividing_reasonable]:
                        new_cellular_arr[segmented_arr == segmented_arr_index] = int(
                            parent_node_this.data.get_number())
                        mapping_cellular_dict[int(segmented_arr_index)] = int(cell_index)
                        dividing_cells.append(parent_node_this.tag)
                        is_found = True
                else:
                    for sx in range(5):
                        for sy in range(5):
                            for sz in range(5):
                                ti1 = segmented_arr[x_tmp + sx, y_tmp + sy, z_tmp + sz]
                                ti2 = segmented_arr[x_tmp - sx, y_tmp - sy, z_tmp - sz]
                                if ti1 != 0 and ti1 not in mapping_cellular_dict:
                                    new_cellular_arr[segmented_arr == ti1] = cell_index
                                    mapping_cellular_dict[int(ti1)] = int(cell_index)
                                    is_found = True
                                    break
                                if ti2 != 0 and ti2 not in mapping_cellular_dict:
                                    new_cellular_arr[segmented_arr == ti2] = cell_index
                                    mapping_cellular_dict[int(ti2)] = int(cell_index)
                                    is_found = True
                                    break
                            else:
                                continue
                            break
                        else:
                            continue
                        break

                if not is_found:
                    lossing_cells.append((label_name_dict.get(cell_index, str(cell_index)),
                                         cell2fate.get(label_name_dict.get(cell_index, ''), 'Unspecified')))
            else:
                new_cellular_arr[segmented_arr == segmented_arr_index] = int(cell_index)
                mapping_cellular_dict[int(segmented_arr_index)] = int(cell_index)

    for cell_index in late_processing_cells:
        nuc_binary_tmp = (annotated_nuc_arr == cell_index)
        locations_cellular = np.where(nuc_binary_tmp)
        centre_len_tmp = 0
        x_tmp = locations_cellular[0][centre_len_tmp]
        y_tmp = locations_cellular[1][centre_len_tmp]
        z_tmp = locations_cellular[2][centre_len_tmp]
        segmented_arr_index = segmented_arr[x_tmp, y_tmp, z_tmp]

        if segmented_arr_index in mapping_cellular_dict.keys() or segmented_arr_index == 0:
            nuc_binary_tmp_eroded = skimage.morphology.binary_erosion(nuc_binary_tmp)
            new_cellular_arr[nuc_binary_tmp_eroded] = cell_index
            lossing_cells.append((label_name_dict.get(cell_index, str(cell_index)),
                                  cell2fate.get(label_name_dict.get(cell_index, ''), 'Unspecified')))
        else:
            mapping_cellular_dict[segmented_arr_index] = cell_index

    nib_save(new_cellular_arr,
             os.path.join(output_saving_path, embryo_name, os.path.basename(segCellniigzpath)))

    dict_saving_path = os.path.join(output_saving_path, 'middle_materials', embryo_name, 'mapping',
                                    f'{embryo_name}_{tp_this_str}_mapping_dict.csv')
    check_folder(dict_saving_path)
    pd.DataFrame.from_dict(mapping_cellular_dict, orient='index').to_csv(dict_saving_path)

    if len(dividing_cells) > 0:
        cp_path = os.path.join(output_saving_path, 'middle_materials', embryo_name, 'dividing',
                               f'{embryo_name}_{tp_this_str}_dividing.csv')
        check_folder(cp_path)
        pd.DataFrame(dividing_cells).to_csv(cp_path)
    if len(lossing_cells) > 0:
        lp_path = os.path.join(output_saving_path, 'middle_materials', embryo_name, 'losing',
                               f'{embryo_name}_{tp_this_str}_losing.csv')
        check_folder(lp_path)
        pd.DataFrame(lossing_cells).to_csv(lp_path)


def main():
    parser = argparse.ArgumentParser(description='CTransformer Step 6: 谱系映射')
    parser.add_argument('--seg-root', required=True,
                        help='Step 3 分割输出根目录 (含 {embryo}/SegCell/ 和 SegMemb/)')
    parser.add_argument('--annotated-root', required=True,
                        help='AnnotatedNuc + CD 文件根目录 (DataSource/RunningDataset)')
    parser.add_argument('--name-dict', required=True,
                        help='name_dictionary.csv 路径')
    parser.add_argument('--cell-fate', default=None,
                        help='CellFate.xls 路径 (可选, 没有则跳过凋亡细胞处理)')
    parser.add_argument('--output', required=True,
                        help='谱系映射输出目录')
    parser.add_argument('--embryos', nargs='+', required=True,
                        help='要处理的 embryo 名称列表')
    parser.add_argument('--workers', type=int, default=None,
                        help='并行 worker 数 (默认 CPU/2)')
    args = parser.parse_args()

    # 读取 name dictionary
    label_name_dict = pd.read_csv(args.name_dict, index_col=0).to_dict()['0']
    name_label_dict = {value: key for key, value in label_name_dict.items()}

    # 读取 cell fate (可选)
    cell2fate = {}
    if args.cell_fate and os.path.exists(args.cell_fate):
        cell_fate = pd.read_excel(args.cell_fate, names=["Cell", "Fate"],
                                  converters={"Cell": str, "Fate": str}, header=None)
        cell_fate = cell_fate.map(lambda x: x[:-1] if isinstance(x, str) and len(x) > 0 else x)
        cell2fate = dict(zip(cell_fate.Cell, cell_fate.Fate))
        print(f'[CellFate] 已加载 {len(cell2fate)} 个细胞命运条目')
    else:
        print('[CellFate] 未提供或不存在, 跳过凋亡细胞特殊处理')

    os.makedirs(args.output, exist_ok=True)

    for embryo_name in args.embryos:
        print(f'\n{"="*60}')
        print(f'  谱系映射: {embryo_name}')
        print(f'{"="*60}')

        # 检查 SegCell 输出
        seg_cell_dir = os.path.join(args.seg_root, embryo_name, 'SegCell')
        seg_memb_dir = os.path.join(args.seg_root, embryo_name, 'SegMemb')
        if not os.path.exists(seg_cell_dir):
            print(f'  [跳过] SegCell 不存在: {seg_cell_dir}')
            continue

        segmented_cell_paths = sorted(glob(os.path.join(seg_cell_dir, '*.nii.gz')), reverse=True)
        if not segmented_cell_paths:
            print(f'  [跳过] SegCell 为空')
            continue

        # 检查 CD 文件
        cd_file = os.path.join(args.annotated_root, embryo_name, f'CD{embryo_name}.csv')
        if not os.path.exists(cd_file):
            # 也检查 "CD FILES" 目录
            cd_file_alt = os.path.join(os.path.dirname(args.annotated_root), 'CD FILES', f'CD{embryo_name}.csv')
            if os.path.exists(cd_file_alt):
                cd_file = cd_file_alt
            else:
                print(f'  [跳过] CD 文件不存在: {cd_file}')
                continue

        max_time = len(segmented_cell_paths)
        print(f'  SegCell: {max_time} 个时间点')
        print(f'  CD 文件: {cd_file}')

        # 构建细胞树
        cell_tree_embryo = construct_celltree(cd_file, max_time, args.name_dict)

        # ==================== Phase 1: 核→细胞映射 ====================
        print(f'  Phase 1: 核标号→细胞映射...')
        parameters = []
        for seg_cell_file in segmented_cell_paths:
            _, tp_this = os.path.basename(seg_cell_file).split('_')[:2]
            annotated_path = os.path.join(args.annotated_root, embryo_name, 'AnnotatedNuc',
                                          f'{embryo_name}_{tp_this}_annotatedNuc.nii.gz')
            if not os.path.exists(annotated_path):
                continue
            parameters.append([seg_cell_file, cell_tree_embryo, annotated_path,
                               embryo_name, cell2fate, label_name_dict, args.output])

        if not parameters:
            print(f'  [跳过] 没有匹配的 AnnotatedNuc 文件')
            continue

        n_workers = args.workers or min(len(parameters), max(mp.cpu_count() // 2, 1))
        print(f'  处理 {len(parameters)} 个时间点, workers={n_workers}')
        pool = mp.Pool(n_workers)
        for _ in tqdm(pool.imap_unordered(running_reassign_cellular_map, parameters),
                      total=len(parameters), desc=f'{embryo_name} 核→细胞'):
            pass
        pool.close()
        pool.join()

        # ==================== Phase 2: 合并分裂细胞 ====================
        print(f'  Phase 2: 合并分裂细胞...')
        reassigned_paths = sorted(glob(os.path.join(args.output, embryo_name, '*.nii.gz')), reverse=True)
        parameters_div = []
        for reassigned_file in reassigned_paths:
            _, tp_this = os.path.basename(reassigned_file).split('_')[:2]
            memb_path = os.path.join(seg_memb_dir, f'{embryo_name}_{tp_this}_segMemb.nii.gz')
            ann_path = os.path.join(args.annotated_root, embryo_name, 'AnnotatedNuc',
                                    f'{embryo_name}_{tp_this}_annotatedNuc.nii.gz')
            if os.path.exists(memb_path) and os.path.exists(ann_path):
                parameters_div.append([reassigned_file, memb_path, ann_path,
                                       cell_tree_embryo, name_label_dict, label_name_dict])

        if parameters_div:
            # 分裂合并用单进程 (共享 cell_tree 状态)
            pool2 = mp.Pool(1)
            for _ in tqdm(pool2.imap_unordered(combine_division_mp, parameters_div),
                          total=len(parameters_div), desc=f'{embryo_name} 合并分裂'):
                pass
            pool2.close()
            pool2.join()

        # 统计
        out_files = glob(os.path.join(args.output, embryo_name, '*.nii.gz'))
        map_files = glob(os.path.join(args.output, 'middle_materials', embryo_name, 'mapping', '*.csv'))
        print(f'  完成! 输出: {len(out_files)} 个 NIfTI, {len(map_files)} 个 mapping CSV')

    print(f'\n全部完成! 输出目录: {args.output}')


if __name__ == '__main__':
    main()
