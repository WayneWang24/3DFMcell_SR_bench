#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cellsr


# python ./data_pre_process/prepare_cells_for_seesr.py \
#   --src-root /root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/fastz_200_highL/cells_xy_p1p99/train \
#   --dest-root /root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/set_seesr/fastz_200_highL/cells_xy_p1p99 \
#   --scales 2,4,8 \
#   --tile 512 --stride 512 \
#   --pad edge \
#   --norm p1p99 \
#   --to-uint8 \
#   --lr-up-interp cubic \
#   --tag-mode empty



# python ./data_pre_process/prepare_cells_for_seesr.py \
#   --src-root /root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/fastz_200_highL/cells_xz_p1p99/train \
#   --dest-root /root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/set_seesr/fastz_200_highL/cells_xz_p1p99 \
#   --scales 2,4,8 \
#   --tile 512 --stride 512 \
#   --pad edge \
#   --norm p1p99 \
#   --to-uint8 \
#   --lr-up-interp cubic \
#   --tag-mode empty

# python ./data_pre_process/prepare_cells_for_seesr.py \
#   --src-root /root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/fastz_200_highL/cells_yz_p1p99/train \
#   --dest-root /root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/set_seesr/fastz_200_highL/cells_yz_p1p99 \
#   --scales 2,4,8 \
#   --tile 512 --stride 512 \
#   --pad edge \
#   --norm p1p99 \
#   --to-uint8 \
#   --lr-up-interp cubic \
#   --tag-mode empty




# python ./data_pre_process/prepare_cells_for_seesr.py \
#   --src-root /root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/singleS_30_highL/cells_xy_p1p99/train \
#   --dest-root /root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/set_seesr/singleS_30_highL/cells_xy_p1p99 \
#   --scales 2,4,8 \
#   --tile 512 --stride 512 \
#   --pad edge \
#   --norm p1p99 \
#   --to-uint8 \
#   --lr-up-interp cubic \
#   --tag-mode empty



# python ./data_pre_process/prepare_cells_for_seesr.py \
#   --src-root /root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/singleS_30_highL/cells_xz_p1p99/train \
#   --dest-root /root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/set_seesr/singleS_30_highL/cells_xz_p1p99 \
#   --scales 2,4,8 \
#   --tile 512 --stride 512 \
#   --pad edge \
#   --norm p1p99 \
#   --to-uint8 \
#   --lr-up-interp cubic \
#   --tag-mode empty


# python ./data_pre_process/prepare_cells_for_seesr.py \
#   --src-root /root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/singleS_30_highL/cells_yz_p1p99/train \
#   --dest-root /root/autodl-tmp/datasets/3dFM_cell_SR_bench_dataset/set_seesr/singleS_30_highL/cells_yz_p1p99 \
#   --scales 2,4,8 \
#   --tile 512 --stride 512 \
#   --pad edge \
#   --norm p1p99 \
#   --to-uint8 \
#   --lr-up-interp cubic \
#   --tag-mode empty



