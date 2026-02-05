#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cellsr


################### esrgan #######################
# # esrgan X4 - singleS_30_highL xy 
# python ../mmagic/tools/train.py ../configs/esrgan_x4_cells_xy_gan.py \
#   --work-dir results/work_dirs/esrgan_x4_singleS_xy_p1p99 \
#   --cfg-options experiment_name=esrgan_x4_singleS_xy_p1p99 \
#                 data_root=/root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/singleS_30_highL/cells_xy_p1p99 \
#                 scale=4

# # esrgan X4 - singleS_30_highL yz 
# python ../mmagic/tools/train.py ../configs/esrgan_x4_cells_xy_gan.py \
#   --work-dir results/work_dirs/esrgan_x4_singleS_yz_p1p99 \
#   --cfg-options experiment_name=esrgan_x4_singleS_yz_p1p99 \
#                 data_root=/root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/singleS_30_highL/cells_yz_p1p99 \
#                 scale=4

# # esrgan X4 - singleS_30_highL xz
# python ../mmagic/tools/train.py ../configs/esrgan_x4_cells_xy_gan.py \
#   --work-dir results/work_dirs/esrgan_x4_singleS_xz_p1p99 \
#   --cfg-options experiment_name=esrgan_x4_singleS_xz_p1p99 \
#                 data_root=/root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/singleS_30_highL/cells_xz_p1p99 \
#                 scale=4

# esrgan X4 - fastz_200_highL xy
python ../mmagic/tools/train.py ../configs/esrgan_x4_cells_xy_gan.py \
  --work-dir results/work_dirs/esrgan_x4_fastz_xy_p1p99_post \
  --cfg-options experiment_name=esrgan_x4_fastz_xy_p1p99_post \
                data_root=/root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/fastz_200_highL/cells_xy_p1p99 \
                scale=4 \
                model.generator.init_cfg.checkpoint=/root/autodl-tmp/codes/3DFMcell_SR_bench/scripts/results/work_dirs/esrgan_x4_fastz_xy_p1p99/iter_100000.pth

# # esrgan X4 - fastz_200_highL yz
# python ../mmagic/tools/train.py ../configs/esrgan_x4_cells_xy_gan.py \
#   --work-dir results/work_dirs/esrgan_x4_fastz_yz_p1p99 \
#   --cfg-options experiment_name=esrgan_x4_fastz_yz_p1p99 \
#                 data_root=/root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/fastz_200_highL/cells_yz_p1p99 \
#                 scale=4

# esrgan X4 - fastz_200_highL xz
python ../mmagic/tools/train.py ../configs/esrgan_x4_cells_xy_gan.py \
  --work-dir results/work_dirs/esrgan_x4_fastz_xz_p1p99_post \
  --cfg-options experiment_name=esrgan_x4_fastz_xz_p1p99_post \
                data_root=/root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/fastz_200_highL/cells_xz_p1p99 \
                scale=4 \
                model.generator.init_cfg.checkpoint=/root/autodl-tmp/codes/3DFMcell_SR_bench/scripts/results/work_dirs/esrgan_x4_fastz_xz_p1p99/iter_100000.pth
