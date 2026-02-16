#!/bin/bash
# =============================================================
# CTransformer 恢复脚本: 补跑 SegCell + 谱系映射
#
# 跳过已完成的 embryo，只处理有 SegMemb 但没有 SegCell 的
# 不需要 GPU（SegCell 是 CPU watershed）
#
# 用法:
#   sudo docker start cellsr_eval_v2 && sudo docker exec -it cellsr_eval_v2 bash
#   cd /home/vm/workspace/3DFMcell_SR_bench
#   bash scripts/run_ct_recovery.sh
# =============================================================
set -e

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
CT_ROOT=$PROJECT_ROOT/CTransformer

# ============================================================
# 路径配置
# ============================================================
DATASOURCE=/home/vm/workspace/DataSource
RUN_DATA=$DATASOURCE/RunningDataset
NAME_DICT=$DATASOURCE/name_dictionary.csv
CELL_FATE=$DATASOURCE/CellFate.xls
CKPT=$CT_ROOT/ckpts/sTUNETr_1_20240203/model_epoch_1000_edt6_sTUNETr.pth

SEG_OUTPUT=$PROJECT_ROOT/scripts/results/ct_full_seg
LINEAGE_OUTPUT=$PROJECT_ROOT/scripts/results/ct_full_lineage

echo "============================================"
echo "  CTransformer 恢复脚本"
echo "============================================"
echo "  分割目录: $SEG_OUTPUT"
echo "  谱系输出: $LINEAGE_OUTPUT"
echo "============================================"

# ============================================================
# Step 0: 安装依赖
# ============================================================
echo ""
echo "============================================"
echo "  Step 0: 检查依赖"
echo "============================================"
python -c "import monai" 2>/dev/null || pip install monai==1.3.0
python -c "import torchmetrics" 2>/dev/null || pip install torchmetrics==0.11.4
python -c "import tiler" 2>/dev/null || pip install tiler
python -c "import treelib" 2>/dev/null || pip install treelib
python -c "import scipy" 2>/dev/null || pip install scipy
python -c "import nibabel" 2>/dev/null || pip install nibabel --no-deps
python -c "import skimage" 2>/dev/null || pip install scikit-image
python -c "import gtda" 2>/dev/null || pip install giotto-tda
python -c "import openpyxl" 2>/dev/null || pip install openpyxl
echo "依赖检查完成"

# ============================================================
# Step 1: 兼容性补丁
# ============================================================
echo ""
echo "============================================"
echo "  Step 1: 应用兼容性补丁"
echo "============================================"

for f in $CT_ROOT/3_run_segmentation.py $CT_ROOT/run_segmentation.py; do
    if [ -f "$f" ] && grep -q "torch.load(" "$f" && ! grep -q "weights_only" "$f"; then
        sed -i 's/torch\.load(\(.*\))/torch.load(\1, weights_only=False)/g' "$f"
        echo "  [patched] torch.load: $(basename $f)"
    fi
done

if [ -f "$CT_ROOT/3_run_segmentation.py" ] && ! grep -q 'k.replace("module."' "$CT_ROOT/3_run_segmentation.py"; then
    sed -i '/check_point = torch.load/a\    check_point["state_dict"] = {k.replace("module.", ""): v for k, v in check_point["state_dict"].items()}' "$CT_ROOT/3_run_segmentation.py"
    echo "  [patched] module. prefix strip"
fi

if [ -f "$CT_ROOT/3_run_segmentation.py" ]; then
    sed -i 's/num_workers=1/num_workers=0/g' "$CT_ROOT/3_run_segmentation.py"
fi

for f in $CT_ROOT/segmentation_utils/ProcessLib.py $CT_ROOT/6_build_cell_shape_map.py $CT_ROOT/data_utils/augmentations.py; do
    if [ -f "$f" ] && grep -q 'np\.float)' "$f"; then
        sed -i 's/np\.float)/float)/g' "$f"
    fi
done

if [ -f "$CT_ROOT/data_utils/transforms.py" ] && grep -q 'collections\.Sequence' "$CT_ROOT/data_utils/transforms.py"; then
    sed -i 's/collections\.Sequence/collections.abc.Sequence/g' "$CT_ROOT/data_utils/transforms.py"
fi

for f in $(grep -rl "scipy.ndimage.morphology" "$CT_ROOT" 2>/dev/null || true); do
    if [ -f "$f" ]; then
        sed -i 's/from scipy\.ndimage\.morphology import/from scipy.ndimage import/g' "$f"
    fi
done

echo "补丁完成"

# ============================================================
# Step 2: 扫描需要补跑 SegCell 的 embryo
# ============================================================
echo ""
echo "============================================"
echo "  Step 2: 扫描 embryo 状态"
echo "============================================"

NEED_SEGCELL=""
DONE_SEGCELL=""
SKIP_NO_MEMB=""

for embryo_dir in "$SEG_OUTPUT"/*/; do
    [ -d "$embryo_dir" ] || continue
    embryo=$(basename "$embryo_dir")

    memb_count=$(ls "$embryo_dir/SegMemb"/*.nii.gz 2>/dev/null | wc -l)
    cell_count=$(ls "$embryo_dir/SegCell"/*.nii.gz 2>/dev/null | wc -l)

    if [ "$memb_count" -eq 0 ]; then
        echo "  [跳过] $embryo: SegMemb=0 (无膜预测结果)"
        SKIP_NO_MEMB="$SKIP_NO_MEMB $embryo"
    elif [ "$cell_count" -ge "$memb_count" ]; then
        echo "  [完成] $embryo: SegMemb=$memb_count, SegCell=$cell_count"
        DONE_SEGCELL="$DONE_SEGCELL $embryo"
    else
        echo "  [待跑] $embryo: SegMemb=$memb_count, SegCell=$cell_count"
        NEED_SEGCELL="$NEED_SEGCELL $embryo"
    fi
done

NEED_SEGCELL=$(echo $NEED_SEGCELL | xargs)  # trim whitespace

if [ -z "$NEED_SEGCELL" ]; then
    echo ""
    echo "  所有 embryo 的 SegCell 已完成，跳到谱系步骤"
else
    NEED_COUNT=$(echo $NEED_SEGCELL | wc -w)
    echo ""
    echo "  需要补跑 SegCell: $NEED_COUNT 个 embryo"
    echo "  列表: $NEED_SEGCELL"

    # ============================================================
    # Step 3: 逐个 embryo 补跑 SegCell
    # ============================================================
    echo ""
    echo "============================================"
    echo "  Step 3: 补跑 SegCell (无需 GPU)"
    echo "============================================"

    SUCCESS=0
    FAILED=""

    for embryo in $NEED_SEGCELL; do
        echo ""
        echo "--- [$embryo] ---"

        # 生成单 embryo 配置
        EMBRYO_PYLIST="['$embryo']"
        CFG_NAME="3_recovery_${embryo}"

        cat > "$CT_ROOT/para_config/${CFG_NAME}.yaml" << YAMLEOF
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
trained_model: ${CKPT}
seed: 1024

run_data_path: ${RUN_DATA}
output_data_path: ${SEG_OUTPUT}
run_transforms:
  Compose([
    Resize((256, 384, 224)),
    NumpyType((np.float32, np.float32))
    ])

is_predict_memb: False
is_segment_cell: True

is_nuc_labelled: False
is_nuc_predicted: False
is_nuc_predicted_and_localmin: False

mem_edt_threshold: 9
YAMLEOF

        cd "$CT_ROOT"
        if python 3_run_segmentation.py -cfg "$CFG_NAME" -gpu 0 2>&1; then
            cell_count=$(ls "$SEG_OUTPUT/$embryo/SegCell"/*.nii.gz 2>/dev/null | wc -l)
            echo "  [$embryo] 完成! SegCell=$cell_count"
            SUCCESS=$((SUCCESS + 1))
        else
            echo "  [$embryo] 失败!"
            FAILED="$FAILED $embryo"
        fi
        cd "$PROJECT_ROOT"

        # 清理临时配置
        rm -f "$CT_ROOT/para_config/${CFG_NAME}.yaml"
    done

    echo ""
    echo "============================================"
    echo "  SegCell 补跑完成"
    echo "============================================"
    echo "  成功: $SUCCESS / $NEED_COUNT"
    if [ -n "$FAILED" ]; then
        echo "  失败: $FAILED"
    fi
fi

# ============================================================
# Step 4: 汇总分割结果
# ============================================================
echo ""
echo "============================================"
echo "  Step 4: 汇总分割结果"
echo "============================================"

ALL_EMBRYOS=""
for embryo_dir in "$SEG_OUTPUT"/*/; do
    [ -d "$embryo_dir" ] || continue
    embryo=$(basename "$embryo_dir")
    memb_count=$(ls "$embryo_dir/SegMemb"/*.nii.gz 2>/dev/null | wc -l)
    cell_count=$(ls "$embryo_dir/SegCell"/*.nii.gz 2>/dev/null | wc -l)
    echo "  $embryo: SegMemb=$memb_count, SegCell=$cell_count"
    if [ "$cell_count" -gt 0 ]; then
        ALL_EMBRYOS="$ALL_EMBRYOS $embryo"
    fi
done

ALL_EMBRYOS=$(echo $ALL_EMBRYOS | xargs)

if [ -z "$ALL_EMBRYOS" ]; then
    echo "  [错误] 没有任何 embryo 有 SegCell 结果"
    exit 1
fi

# ============================================================
# Step 5: 运行谱系映射
# ============================================================
echo ""
echo "============================================"
echo "  Step 5: 运行谱系映射"
echo "============================================"

cd "$PROJECT_ROOT"

FATE_ARG=""
if [ -f "$CELL_FATE" ]; then
    FATE_ARG="--cell-fate $CELL_FATE"
    echo "  CellFate: $CELL_FATE"
else
    echo "  [注意] CellFate.xls 不存在, 跳过凋亡特殊处理"
fi

echo "  谱系 embryo: $ALL_EMBRYOS"

python "$PROJECT_ROOT/scripts/run_lineage_mapping.py" \
    --seg-root "$SEG_OUTPUT" \
    --annotated-root "$RUN_DATA" \
    --name-dict "$NAME_DICT" \
    $FATE_ARG \
    --output "$LINEAGE_OUTPUT" \
    --embryos $ALL_EMBRYOS

# ============================================================
# 完成
# ============================================================
echo ""
echo "============================================"
echo "  恢复 Pipeline 完成!"
echo "============================================"
echo "  分割输出:  $SEG_OUTPUT"
echo "  谱系输出:  $LINEAGE_OUTPUT"
