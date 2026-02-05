
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cellsr


# SWINR X4 - fastz_200_highL xz
python ../mmagic/tools/test.py \
  ../configs/swin_test_tifR_override.py \
  ./results/work_dirs/swinir_x4_xy_fastz_200_highL_XY_p1p99/iter_100000.pth \
  --work-dir ./results/work_dirs/swinir_x4_xy_fastz_200_highL_XY_p1p99_test \
  --out ./results/work_dirs/swinir_x4_xy_fastz_200_highL_XY_p1p99/swinir_beforedeconv_xy_metrics_psnr_ssim.json
