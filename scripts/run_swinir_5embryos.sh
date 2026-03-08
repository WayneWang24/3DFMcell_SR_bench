#!/bin/bash
# 只跑 SwinIR 的 5 个胚胎: 完整流程 Stage 1-5
set -e

EMBRYOS="190314plc1p3 190222plc1p1 170614plc1p1 200117plc1pop1ip2 181210plc1p3"
GPU_ID=${1:-0}

PROJECT_ROOT=/home/vm/workspace/3DFMcell_SR_bench
CT_ROOT=$PROJECT_ROOT/CTransformer
DATASOURCE_RD=/home/vm/workspace/DataSource/RunningDataset
NAME_DICT=/home/vm/workspace/DataSource/name_dictionary.csv
CELL_FATE=/home/vm/workspace/DataSource/CellFate.xls
EXP2=$PROJECT_ROOT/scripts/results/exp2_swinir

CONFIG=$PROJECT_ROOT/configs/swinir_x4_cells_xy.py
CKPT=$PROJECT_ROOT/scripts/results/work_dirs/swinir_x4_xy_fastz_200_highL_XY_p1p99/best_PSNR_iter_15000.pth
CKPT_CT=$CT_ROOT/ckpts/sTUNETr_1_20240203/model_epoch_1000_edt6_sTUNETr.pth

echo "===== SwinIR 5 胚胎 Pipeline ====="
echo "GPU: $GPU_ID"
echo "胚胎: $EMBRYOS"

# ============================================================
# Step 0: 兼容性补丁
# ============================================================
echo ""
echo "=== Step 0: 兼容性补丁 ==="

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
        print(f'[fix] Patched Adafactor: {p}')
PYEOF

for f in $CT_ROOT/3_run_segmentation.py $CT_ROOT/run_segmentation.py; do
    if [ -f "$f" ] && grep -q "torch.load(" "$f" && ! grep -q "weights_only" "$f"; then
        sed -i 's/torch\.load(\(.*\))/torch.load(\1, weights_only=False)/g' "$f"
    fi
done
if [ -f "$CT_ROOT/3_run_segmentation.py" ] && ! grep -q 'k.replace("module."' "$CT_ROOT/3_run_segmentation.py"; then
    sed -i '/check_point = torch.load/a\    check_point["state_dict"] = {k.replace("module.", ""): v for k, v in check_point["state_dict"].items()}' "$CT_ROOT/3_run_segmentation.py"
fi
[ -f "$CT_ROOT/3_run_segmentation.py" ] && sed -i 's/num_workers=1/num_workers=0/g' "$CT_ROOT/3_run_segmentation.py"
for f in $CT_ROOT/segmentation_utils/ProcessLib.py $CT_ROOT/6_build_cell_shape_map.py $CT_ROOT/data_utils/augmentations.py; do
    [ -f "$f" ] && grep -q 'np\.float)' "$f" && sed -i 's/np\.float)/float)/g' "$f"
done
[ -f "$CT_ROOT/data_utils/transforms.py" ] && grep -q 'collections\.Sequence' "$CT_ROOT/data_utils/transforms.py" && \
    sed -i 's/collections\.Sequence/collections.abc.Sequence/g' "$CT_ROOT/data_utils/transforms.py"
for f in $(grep -rl "scipy.ndimage.morphology" "$CT_ROOT" 2>/dev/null || true); do
    [ -f "$f" ] && sed -i 's/from scipy\.ndimage\.morphology import/from scipy.ndimage import/g' "$f"
done
echo "补丁完成"

# ============================================================
# Stage 1: SR (只跑缺失的胚胎)
# ============================================================
echo ""
echo "=== Stage 1: SR ==="

for embryo in $EMBRYOS; do
    SR_OUT=$EXP2/sr_output/$embryo
    INPUT_DIR=$DATASOURCE_RD/$embryo/RawMemb
    mkdir -p "$SR_OUT"

    DONE_COUNT=$(find "$SR_OUT" -maxdepth 1 -name "*.nii.gz" 2>/dev/null | wc -l | tr -d ' ')
    INPUT_COUNT=$(find "$INPUT_DIR" -maxdepth 1 -name "*rawMemb*.nii.gz" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$DONE_COUNT" -ge "$INPUT_COUNT" ] && [ "$INPUT_COUNT" -gt 0 ]; then
        echo "  [跳过] $embryo: SR 已完成 ($DONE_COUNT/$INPUT_COUNT)"
        continue
    fi

    echo "  [处理] $embryo ($DONE_COUNT/$INPUT_COUNT)"
    python "$PROJECT_ROOT/scripts/eval/sr_3d_nifti.py" \
        --input-dir "$INPUT_DIR" \
        --output-dir "$SR_OUT" \
        --config "$CONFIG" \
        --checkpoint "$CKPT" \
        --scale 4 --slice-axis xz \
        --device "cuda:$GPU_ID"

    # 提升子目录文件
    if ls "$SR_OUT"/*/*.nii.gz 1>/dev/null 2>&1; then
        mv "$SR_OUT"/*/*.nii.gz "$SR_OUT/" 2>/dev/null || true
        rmdir "$SR_OUT"/*/ 2>/dev/null || true
    fi
    echo "  [完成] $embryo"
done
echo "Stage 1 完成"

# ============================================================
# Stage 2: 强度修复
# ============================================================
echo ""
echo "=== Stage 2: 强度修复 ==="

for embryo in $EMBRYOS; do
    SR_DIR=$EXP2/sr_output/$embryo
    FIXED_DIR=$EXP2/sr_fixed/$embryo
    [ ! -d "$SR_DIR" ] && continue

    SR_COUNT=$(find "$SR_DIR" -maxdepth 1 -name "*.nii.gz" 2>/dev/null | wc -l | tr -d ' ')
    FIXED_COUNT=$(find "$FIXED_DIR" -maxdepth 1 -name "*.nii.gz" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$FIXED_COUNT" -ge "$SR_COUNT" ] && [ "$SR_COUNT" -gt 0 ]; then
        echo "  [跳过] $embryo ($FIXED_COUNT/$SR_COUNT)"
        continue
    fi

    echo "  [处理] $embryo"
    python "$PROJECT_ROOT/scripts/fix_sr_intensity.py" \
        --sr-dir "$SR_DIR" \
        --raw-dir "$DATASOURCE_RD/$embryo/RawMemb" \
        --output-dir "$EXP2/sr_fixed"
done
echo "Stage 2 完成"

# ============================================================
# Stage 3: 准备 CTransformer 输入
# ============================================================
echo ""
echo "=== Stage 3: 准备 CTransformer 输入 ==="

python "$PROJECT_ROOT/scripts/prepare_ct_data_datasource.py" \
    --sr-dir "$EXP2/sr_fixed" \
    --datasource "$DATASOURCE_RD" \
    --output-dir "$EXP2/ct_input" \
    --target-size 256 384 224 \
    --embryos $EMBRYOS

echo "Stage 3 完成"

# 清理 sr_fixed
if [ -d "$EXP2/sr_fixed" ]; then
    SIZE=$(du -sh "$EXP2/sr_fixed" 2>/dev/null | cut -f1)
    echo "  [清理] sr_fixed ($SIZE)"
    rm -rf "$EXP2/sr_fixed"
fi

# ============================================================
# Stage 4: CTransformer 分割 (SegMemb + SegCell)
# ============================================================
echo ""
echo "=== Stage 4: CTransformer 分割 ==="

# 生成 YAML 配置
EMBRYO_PYLIST=$(echo $EMBRYOS | tr ' ' '\n' | sed "s/.*/'&'/" | paste -sd, -)
EMBRYO_PYLIST="[$EMBRYO_PYLIST]"

mkdir -p "$CT_ROOT/para_config"
cat > "$CT_ROOT/para_config/3_exp2_swinir_5.yaml" << YAMLEOF
workflow_step: 3_RUN

net: SwinUNETR
net_params:
  img_size: (128,128,128)
  in_channels: 2
  feature_size: 48
  out_channels: 2

is_input_nuc_channel: True
running_embryo_names: ${EMBRYO_PYLIST}
dataset_name: NiigzRunDataset
trained_model: ${CKPT_CT}
seed: 1024

run_data_path: ${EXP2}/ct_input
output_data_path: ${EXP2}/seg
run_transforms:
  Compose([
    Resize((256, 384, 224)),
    NumpyType((np.float32, np.float32))
    ])

is_predict_memb: True
is_segment_cell: True

is_nuc_labelled: False
is_nuc_predicted: False
is_nuc_predicted_and_localmin: False

mem_edt_threshold: 9
YAMLEOF

echo "  SegMemb (GPU)..."
cd "$CT_ROOT"
python 3_run_segmentation.py -cfg 3_exp2_swinir_5 -gpu $GPU_ID
cd "$PROJECT_ROOT"

# 补跑 SegCell (watershed 可能超时)
echo "  补跑 SegCell (32 workers)..."
cd "$CT_ROOT"
python "$PROJECT_ROOT/scripts/resume_segcell.py" \
    --seg-root "$EXP2/seg" \
    --embryos $EMBRYOS \
    --timeout 1800 \
    --workers 32
cd "$PROJECT_ROOT"

echo "Stage 4 完成"

# 清理 ct_input
if [ -d "$EXP2/ct_input" ]; then
    SIZE=$(du -sh "$EXP2/ct_input" 2>/dev/null | cut -f1)
    echo "  [清理] ct_input ($SIZE)"
    rm -rf "$EXP2/ct_input"
fi

# ============================================================
# Stage 5: 谱系映射
# ============================================================
echo ""
echo "=== Stage 5: 谱系映射 ==="

FATE_ARG=""
[ -f "$CELL_FATE" ] && FATE_ARG="--cell-fate $CELL_FATE"

python "$PROJECT_ROOT/scripts/run_lineage_mapping.py" \
    --seg-root "$EXP2/seg" \
    --annotated-root "$DATASOURCE_RD" \
    --name-dict "$NAME_DICT" \
    $FATE_ARG \
    --output "$EXP2/lineage" \
    --embryos $EMBRYOS

echo ""
echo "===== 全部完成! ====="
echo "  seg:     $EXP2/seg/"
echo "  lineage: $EXP2/lineage/"
