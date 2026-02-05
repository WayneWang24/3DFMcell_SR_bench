#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cellsr


################### esrgan #######################
# # esrgan X4 - singleS_30_highL xy 
# python ../mmagic/tools/train.py ../configs/esrgan_psnr_x4_cells_pretrain.py \
#   --work-dir results/work_dirs/esrgan_x4_singleS_xy_p1p99 \
#   --cfg-options experiment_name=esrgan_x4_singleS_xy_p1p99 \
#                 data_root=/root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/singleS_30_highL/cells_xy_p1p99 \
#                 scale=4

# # esrgan X4 - singleS_30_highL yz 
# python ../mmagic/tools/train.py ../configs/esrgan_psnr_x4_cells_pretrain.py \
#   --work-dir results/work_dirs/esrgan_x4_singleS_yz_p1p99 \
#   --cfg-options experiment_name=esrgan_x4_singleS_yz_p1p99 \
#                 data_root=/root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/singleS_30_highL/cells_yz_p1p99 \
#                 scale=4

# # esrgan X4 - singleS_30_highL xz
# python ../mmagic/tools/train.py ../configs/esrgan_psnr_x4_cells_pretrain.py \
#   --work-dir results/work_dirs/esrgan_x4_singleS_xz_p1p99 \
#   --cfg-options experiment_name=esrgan_x4_singleS_xz_p1p99 \
#                 data_root=/root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/singleS_30_highL/cells_xz_p1p99 \
#                 scale=4

# esrgan X4 - fastz_200_highL xy
python ../mmagic/tools/train.py ../configs/esrgan_psnr_x4_cells_pretrain.py \
  --work-dir results/work_dirs/esrgan_x4_fastz_xy_p1p99 \
  --cfg-options experiment_name=esrgan_x4_fastz_xy_p1p99 \
                data_root=/root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/fastz_200_highL/cells_xy_p1p99 \
                scale=4

# # esrgan X4 - fastz_200_highL yz
# python ../mmagic/tools/train.py ../configs/esrgan_psnr_x4_cells_pretrain.py \
#   --work-dir results/work_dirs/esrgan_x4_fastz_yz_p1p99 \
#   --cfg-options experiment_name=esrgan_x4_fastz_yz_p1p99 \
#                 data_root=/root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/fastz_200_highL/cells_yz_p1p99 \
#                 scale=4

# esrgan X4 - fastz_200_highL xz
python ../mmagic/tools/train.py ../configs/esrgan_psnr_x4_cells_pretrain.py \
  --work-dir results/work_dirs/esrgan_x4_fastz_xz_p1p99 \
  --cfg-options experiment_name=esrgan_x4_fastz_xz_p1p99 \
                data_root=/root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/fastz_200_highL/cells_xz_p1p99 \
                scale=4
