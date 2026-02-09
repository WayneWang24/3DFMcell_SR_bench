#!/bin/bash
# ==========================================================
#  Zero-Shot Cross-Condition Generalization (ZSCG) 评测
#
#  条件定义:
#    A = deconv (TIF)       — 待处理
#    B = single_shot (NIfTI) — 仅 no-reference 指标
#    C = fastz (NIfTI)       — full-reference + no-reference
#
#  本脚本评测跨条件泛化性能:
#    C→B : fastz 模型 → single_shot 数据 (no-ref only)
#    A→B : deconv 模型 → single_shot 数据 (no-ref only)  [待A就绪]
#    C→A : fastz 模型 → deconv 数据 (full-ref + no-ref) [待A就绪]
#    A→C : deconv 模型 → fastz 数据 (full-ref + no-ref) [待A就绪]
#
#  从 scripts/ 目录运行:  bash run_cross_eval.sh
# ==========================================================

set +e

# ---- 修复 mmengine Adafactor 注册冲突 ----
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
DATA_B=../data/cell_dataset/single_shot_volumes   # B = single_shot
DATA_C=../data/cell_dataset/fastz_volumes          # C = fastz
# DATA_A=../data/cell_dataset/deconv               # A = deconv (待处理)
EVAL_2D=eval/batch_eval_sr.py
OUT=results/eval_outputs/cross_condition
SCALE=4

mkdir -p ${OUT}

# ==========================================================
#  C → B : fastz 模型 → single_shot 数据
#  B 无 cross-domain GT，仅计算 no-reference 指标
# ==========================================================
echo ""
echo "######################################################"
echo "#  C → B : fastz models → single_shot data           #"
echo "######################################################"

# --- C→B : XY ---
echo ""
echo ">>> C→B / xy"
if [ -f ${OUT}/cross_C2B_xy.csv ]; then echo "[跳过] cross_C2B_xy.csv 已存在"; else
python ${EVAL_2D} \
  --model-dirs \
    ${WD}/edsr_x4_cells_fastz_xy_p1p99 \
    ${WD}/swinir_x4_xy_fastz_200_highL_XY_p1p99 \
    ${WD}/esrgan_x4_fastz_xy_p1p99 \
    ${WD}/esrgan_x4_fastz_xy_p1p99_post \
  --data-roots ${DATA_B}/cells_xy_p1p99/val \
  --scales ${SCALE} \
  --output ${OUT}/cross_C2B_xy.csv
fi

# --- C→B : YZ ---
echo ""
echo ">>> C→B / yz"
if [ -f ${OUT}/cross_C2B_yz.csv ]; then echo "[跳过] cross_C2B_yz.csv 已存在"; else
python ${EVAL_2D} \
  --model-dirs \
    ${WD}/srcnn_x4_fastz_yz_p1p99 \
    ${WD}/swinir_x4_fastz_yz_p1p99 \
  --data-roots ${DATA_B}/cells_yz_p1p99/val \
  --scales ${SCALE} \
  --output ${OUT}/cross_C2B_yz.csv
fi

# --- C→B : XZ ---
echo ""
echo ">>> C→B / xz"
if [ -f ${OUT}/cross_C2B_xz.csv ]; then echo "[跳过] cross_C2B_xz.csv 已存在"; else
python ${EVAL_2D} \
  --model-dirs \
    ${WD}/srcnn_x4_fastz_xz_p1p99 \
    ${WD}/swinir_x4_fastz_xz_p1p99 \
    ${WD}/edsr_x4_fastz_xz_p1p99 \
    ${WD}/esrgan_x4_fastz_xz_p1p99 \
    ${WD}/esrgan_x4_fastz_xz_p1p99_post \
  --data-roots ${DATA_B}/cells_xz_p1p99/val \
  --scales ${SCALE} \
  --output ${OUT}/cross_C2B_xz.csv
fi

# ==========================================================
#  B → C : single_shot 模型 → fastz 数据
# ==========================================================
echo ""
echo "######################################################"
echo "#  B → C : single_shot models → fastz data           #"
echo "######################################################"

# --- B→C : XY ---
echo ""
echo ">>> B→C / xy"
if [ -f ${OUT}/cross_B2C_xy.csv ]; then echo "[跳过] cross_B2C_xy.csv 已存在"; else
python ${EVAL_2D} \
  --model-dirs \
    ${WD}/edsr_x4_singleS_xy_p1p99 \
    ${WD}/swinir_x4_singleS_30_highL_xy \
    ${WD}/esrgan_x4_singleS_xy_p1p99 \
    ${WD}/esrgan_x4_singleS_xy_p1p99_post \
  --data-roots ${DATA_C}/cells_xy_p1p99/val \
  --scales ${SCALE} \
  --output ${OUT}/cross_B2C_xy.csv
fi

# --- B→C : YZ ---
echo ""
echo ">>> B→C / yz"
if [ -f ${OUT}/cross_B2C_yz.csv ]; then echo "[跳过] cross_B2C_yz.csv 已存在"; else
python ${EVAL_2D} \
  --model-dirs \
    ${WD}/srcnn_x4_singleS_yz_p1p99 \
    ${WD}/swinir_x4_singleS_yz_p1p99 \
  --data-roots ${DATA_C}/cells_yz_p1p99/val \
  --scales ${SCALE} \
  --output ${OUT}/cross_B2C_yz.csv
fi

# --- B→C : XZ ---
echo ""
echo ">>> B→C / xz"
if [ -f ${OUT}/cross_B2C_xz.csv ]; then echo "[跳过] cross_B2C_xz.csv 已存在"; else
python ${EVAL_2D} \
  --model-dirs \
    ${WD}/srcnn_x4_singleS_xz_p1p99 \
    ${WD}/swinir_x4_singleS_xz_p1p99 \
    ${WD}/edsr_x4_singleS_xz_p1p99 \
    ${WD}/esrgan_x4_singleS_xz_p1p99 \
    ${WD}/esrgan_x4_singleS_xz_p1p99_post \
  --data-roots ${DATA_C}/cells_xz_p1p99/val \
  --scales ${SCALE} \
  --output ${OUT}/cross_B2C_xz.csv
fi

# ==========================================================
#  合并跨条件结果
# ==========================================================
echo ""
echo ">>> 合并所有跨条件评测结果"
python -c "
import pandas as pd, glob, os
csvs = sorted(glob.glob('${OUT}/cross_*.csv'))
if csvs:
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    out = '${OUT}/cross_all.csv'
    df.to_csv(out, index=False, float_format='%.4f')
    print(f'合并 {len(csvs)} 个 CSV → {out}  ({len(df)} 条记录)')
    print(df.to_string(index=False))
else:
    print('无 CSV 文件可合并')
"

echo ""
echo "======================================================"
echo "  跨条件评测完成！结果保存在: ${OUT}/"
echo "======================================================"
