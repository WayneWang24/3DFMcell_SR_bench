#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cellsr


################### swinir #######################

# SWINR X4 - singleS_30_highL yz 
python ../mmagic/tools/train.py ../configs/swinir_x4_cells_xy.py \
  --work-dir results/work_dirs/swinir_x4_singleS_yz_p1p99 \
  --cfg-options experiment_name=swinir_x4_singleS_yz_p1p99 \
                data_root=/root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/singleS_30_highL/cells_yz_p1p99 \
                scale=4

# SWINR X4 - singleS_30_highL xz
python ../mmagic/tools/train.py ../configs/swinir_x4_cells_xy.py \
  --work-dir results/work_dirs/swinir_x4_singleS_xz_p1p99 \
  --cfg-options experiment_name=swinir_x4_singleS_xz_p1p99 \
                data_root=/root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/singleS_30_highL/cells_xz_p1p99 \
                scale=4

# SWINR X4 - fastz_200_highL yz
python ../mmagic/tools/train.py ../configs/swinir_x4_cells_xy.py \
  --work-dir results/work_dirs/swinir_x4_fastz_yz_p1p99 \
  --cfg-options experiment_name=swinir_x4_fastz_yz_p1p99 \
                data_root=/root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/fastz_200_highL/cells_yz_p1p99 \
                scale=4

# SWINR X4 - fastz_200_highL xz
python ../mmagic/tools/train.py ../configs/swinir_x4_cells_xy.py \
  --work-dir results/work_dirs/swinir_x4_fastz_xz_p1p99 \
  --cfg-options experiment_name=swinir_x4_fastz_xz_p1p99 \
                data_root=/root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/fastz_200_highL/cells_xz_p1p99 \
                scale=4
