#!/bin/bash
# ==========================================================
#  批量评测所有训练好的 SR 模型
#  Part 1: 2D 评测 (batch_eval_sr.py) — 推理 + 指标计算
#  Part 2: 3D NIfTI 评测 (sr_3d_nifti.py) — 逐切片超分重建
#
#  从 scripts/ 目录运行:  bash run_all_eval.sh
# ==========================================================

set -e

# ---- 修复 mmengine Adafactor 注册冲突（仅需执行一次）----
python3 << 'PYEOF'
import glob
for p in glob.glob('/opt/conda/lib/python*/site-packages/mmengine/optim/optimizer/builder.py'):
    with open(p) as f:
        s = f.read()
    old = "OPTIMIZERS.register_module(name='Adafactor', module=Adafactor)"
    new = "OPTIMIZERS.register_module(name='Adafactor', module=Adafactor, force=True)"
    if old in s:
        with open(p, 'w') as f:
            f.write(s.replace(old, new))
        print(f'[fix] Patched Adafactor conflict: {p}')
    else:
        print(f'[ok] Already patched: {p}')
PYEOF

# ---- 环境激活（Docker 中可跳过）----
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate cellsr
fi

# ---- 路径配置 ----
WD=results/work_dirs
DATA=../data/cell_dataset
EVAL_2D=eval/batch_eval_sr.py
EVAL_3D=eval/sr_3d_nifti.py
OUT=results/eval_outputs
SCALE=4

NII_SINGLES=/home/vm/workspace/3DFMcell_SR_bench/data/raw_cell_datasets/single_shot_volumes
NII_FASTZ=/home/vm/workspace/3DFMcell_SR_bench/data/raw_cell_datasets/fastz_volumes

mkdir -p ${OUT}

# ==========================================================
#  Part 1: 2D Batch Evaluation
# ==========================================================
echo ""
echo "######################################################"
echo "#  Part 1: 2D Batch Evaluation (batch_eval_sr.py)    #"
echo "######################################################"

# ----------------------------------------------------------
#  single_shot_volumes — XY plane
# ----------------------------------------------------------
echo ""
echo ">>> single_shot / xy"
python ${EVAL_2D} \
  --model-dirs \
    ${WD}/swinir_x4_singleS_30_highL_xy \
    ${WD}/edsr_x4_singleS_xy_p1p99 \
    ${WD}/esrgan_x4_singleS_xy_p1p99 \
    ${WD}/esrgan_x4_singleS_xy_p1p99_post \
  --data-roots ${DATA}/single_shot_volumes/cells_xy_p1p99/val \
  --scales ${SCALE} \
  --output ${OUT}/eval_singleS_xy.csv

# ----------------------------------------------------------
#  single_shot_volumes — YZ plane
# ----------------------------------------------------------
echo ""
echo ">>> single_shot / yz"
python ${EVAL_2D} \
  --model-dirs \
    ${WD}/srcnn_x4_singleS_yz_p1p99 \
    ${WD}/swinir_x4_singleS_yz_p1p99 \
  --data-roots ${DATA}/single_shot_volumes/cells_yz_p1p99/val \
  --scales ${SCALE} \
  --output ${OUT}/eval_singleS_yz.csv

# ----------------------------------------------------------
#  single_shot_volumes — XZ plane
# ----------------------------------------------------------
echo ""
echo ">>> single_shot / xz"
python ${EVAL_2D} \
  --model-dirs \
    ${WD}/srcnn_x4_singleS_xz_p1p99 \
    ${WD}/swinir_x4_singleS_xz_p1p99 \
    ${WD}/edsr_x4_singleS_xz_p1p99 \
    ${WD}/esrgan_x4_singleS_xz_p1p99 \
    ${WD}/esrgan_x4_singleS_xz_p1p99_post \
  --data-roots ${DATA}/single_shot_volumes/cells_xz_p1p99/val \
  --scales ${SCALE} \
  --output ${OUT}/eval_singleS_xz.csv

# ----------------------------------------------------------
#  fastz_volumes — XY plane
# ----------------------------------------------------------
echo ""
echo ">>> fastz / xy"
python ${EVAL_2D} \
  --model-dirs \
    ${WD}/swinir_x4_xy_fastz_200_highL_XY_p1p99 \
    ${WD}/edsr_x4_cells_fastz_xy_p1p99 \
    ${WD}/esrgan_x4_fastz_xy_p1p99 \
    ${WD}/esrgan_x4_fastz_xy_p1p99_post \
  --data-roots ${DATA}/fastz_volumes/cells_xy_p1p99/val \
  --scales ${SCALE} \
  --output ${OUT}/eval_fastz_xy.csv

# ----------------------------------------------------------
#  fastz_volumes — YZ plane
# ----------------------------------------------------------
echo ""
echo ">>> fastz / yz"
python ${EVAL_2D} \
  --model-dirs \
    ${WD}/srcnn_x4_fastz_yz_p1p99 \
    ${WD}/swinir_x4_fastz_yz_p1p99 \
  --data-roots ${DATA}/fastz_volumes/cells_yz_p1p99/val \
  --scales ${SCALE} \
  --output ${OUT}/eval_fastz_yz.csv

# ----------------------------------------------------------
#  fastz_volumes — XZ plane
# ----------------------------------------------------------
echo ""
echo ">>> fastz / xz"
python ${EVAL_2D} \
  --model-dirs \
    ${WD}/srcnn_x4_fastz_xz_p1p99 \
    ${WD}/swinir_x4_fastz_xz_p1p99 \
    ${WD}/edsr_x4_fastz_xz_p1p99 \
    ${WD}/esrgan_x4_fastz_xz_p1p99 \
    ${WD}/esrgan_x4_fastz_xz_p1p99_post \
  --data-roots ${DATA}/fastz_volumes/cells_xz_p1p99/val \
  --scales ${SCALE} \
  --output ${OUT}/eval_fastz_xz.csv

# ----------------------------------------------------------
#  合并所有 2D 评测 CSV
# ----------------------------------------------------------
echo ""
echo ">>> 合并所有 2D 评测结果"
python -c "
import pandas as pd, glob, os
csvs = sorted(glob.glob('${OUT}/eval_*.csv'))
if csvs:
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    out = '${OUT}/eval_all_2d.csv'
    df.to_csv(out, index=False, float_format='%.4f')
    print(f'合并 {len(csvs)} 个 CSV → {out}  ({len(df)} 条记录)')
    print(df.to_string(index=False))
else:
    print('无 CSV 文件可合并')
"

# ==========================================================
#  Part 2: 3D NIfTI Super-Resolution (sr_3d_nifti.py)
#  需要原始 .nii.gz 体积，取消注释并设置 NII 路径后使用
# ==========================================================
echo ""
echo "######################################################"
echo "#  Part 2: 3D NIfTI Evaluation (sr_3d_nifti.py)      #"
echo "######################################################"

if [ -z "${NII_SINGLES}" ] && [ -z "${NII_FASTZ}" ]; then
  echo "跳过 3D 评测：未设置 NII_SINGLES / NII_FASTZ 路径"
  echo "如需启用，请编辑脚本顶部的 NII 路径变量"
else

  # --- single_shot: XZ 模型 (沿 Y 切, Z 轴超分) ---
  if [ -n "${NII_SINGLES}" ]; then
    echo ""
    echo ">>> 3D single_shot — xz slice axis"
    python ${EVAL_3D} \
      --input-dir ${NII_SINGLES} \
      --output-dir ${OUT}/sr_3d_singleS_xz \
      --config \
        ${WD}/srcnn_x4_singleS_xz_p1p99/srcnn_x4_cells_xy_p1p99.py \
        ${WD}/swinir_x4_singleS_xz_p1p99/swinir_x4_cells_xy.py \
        ${WD}/edsr_x4_singleS_xz_p1p99/edsr_x_cells_xy_template.py \
        ${WD}/esrgan_x4_singleS_xz_p1p99/esrgan_psnr_x4_cells_pretrain.py \
        ${WD}/esrgan_x4_singleS_xz_p1p99_post/esrgan_x4_cells_xy_gan.py \
      --checkpoint \
        ${WD}/srcnn_x4_singleS_xz_p1p99/best_PSNR_iter_95000.pth \
        ${WD}/swinir_x4_singleS_xz_p1p99/best_PSNR_iter_10000.pth \
        ${WD}/edsr_x4_singleS_xz_p1p99/best_PSNR_iter_15000.pth \
        ${WD}/esrgan_x4_singleS_xz_p1p99/best_PSNR_iter_15000.pth \
        ${WD}/esrgan_x4_singleS_xz_p1p99_post/best_PSNR_iter_35000.pth \
      --scale ${SCALE} --slice-axis xz

    # --- single_shot: YZ 模型 (沿 X 切, Z 轴超分) ---
    echo ""
    echo ">>> 3D single_shot — yz slice axis"
    python ${EVAL_3D} \
      --input-dir ${NII_SINGLES} \
      --output-dir ${OUT}/sr_3d_singleS_yz \
      --config \
        ${WD}/srcnn_x4_singleS_yz_p1p99/srcnn_x4_cells_xy_p1p99.py \
        ${WD}/swinir_x4_singleS_yz_p1p99/swinir_x4_cells_xy.py \
      --checkpoint \
        ${WD}/srcnn_x4_singleS_yz_p1p99/best_PSNR_iter_95000.pth \
        ${WD}/swinir_x4_singleS_yz_p1p99/best_PSNR_iter_45000.pth \
      --scale ${SCALE} --slice-axis yz
  fi

  # --- fastz: XZ 模型 ---
  if [ -n "${NII_FASTZ}" ]; then
    echo ""
    echo ">>> 3D fastz — xz slice axis"
    python ${EVAL_3D} \
      --input-dir ${NII_FASTZ} \
      --output-dir ${OUT}/sr_3d_fastz_xz \
      --config \
        ${WD}/srcnn_x4_fastz_xz_p1p99/srcnn_x4_cells_xy_p1p99.py \
        ${WD}/swinir_x4_fastz_xz_p1p99/swinir_x4_cells_xy.py \
        ${WD}/edsr_x4_fastz_xz_p1p99/edsr_x_cells_xy_template.py \
        ${WD}/esrgan_x4_fastz_xz_p1p99/esrgan_psnr_x4_cells_pretrain.py \
        ${WD}/esrgan_x4_fastz_xz_p1p99_post/esrgan_x4_cells_xy_gan.py \
      --checkpoint \
        ${WD}/srcnn_x4_fastz_xz_p1p99/best_PSNR_iter_75000.pth \
        ${WD}/swinir_x4_fastz_xz_p1p99/best_PSNR_iter_25000.pth \
        ${WD}/edsr_x4_fastz_xz_p1p99/best_PSNR_iter_10000.pth \
        ${WD}/esrgan_x4_fastz_xz_p1p99/best_PSNR_iter_85000.pth \
        ${WD}/esrgan_x4_fastz_xz_p1p99_post/best_PSNR_iter_35000.pth \
      --scale ${SCALE} --slice-axis xz

    # --- fastz: YZ 模型 ---
    echo ""
    echo ">>> 3D fastz — yz slice axis"
    python ${EVAL_3D} \
      --input-dir ${NII_FASTZ} \
      --output-dir ${OUT}/sr_3d_fastz_yz \
      --config \
        ${WD}/srcnn_x4_fastz_yz_p1p99/srcnn_x4_cells_xy_p1p99.py \
        ${WD}/swinir_x4_fastz_yz_p1p99/swinir_x4_cells_xy.py \
      --checkpoint \
        ${WD}/srcnn_x4_fastz_yz_p1p99/best_PSNR_iter_100000.pth \
        ${WD}/swinir_x4_fastz_yz_p1p99/best_PSNR_iter_20000.pth \
      --scale ${SCALE} --slice-axis yz
  fi

fi

echo ""
echo "======================================================"
echo "  评测完成！结果保存在: ${OUT}/"
echo "======================================================"
