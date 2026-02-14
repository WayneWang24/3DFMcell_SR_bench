#!/bin/bash
# =============================================================
# 完整 CTransformer Pipeline: 分割 + 谱系映射
#
# 在 polyHPC Docker 容器内运行:
#   sudo docker start cellsr_eval_v2 && sudo docker exec -it cellsr_eval_v2 bash
#   cd /home/vm/workspace/3DFMcell_SR_bench
#   bash scripts/run_full_ct_pipeline.sh [embryo1 embryo2 ...]
#
# 默认跑 2 个胚胎做测试，可传参指定:
#   bash scripts/run_full_ct_pipeline.sh 170704plc1p1 200113plc1p2
#   bash scripts/run_full_ct_pipeline.sh ALL   # 跑全部
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
CELL_FATE=$DATASOURCE/CellFate.xls    # 可能不存在，脚本会自动处理
CKPT=$CT_ROOT/ckpts/sTUNETr_1_20240203/model_epoch_1000_edt6_sTUNETr.pth

SEG_OUTPUT=$PROJECT_ROOT/scripts/results/ct_full_seg
LINEAGE_OUTPUT=$PROJECT_ROOT/scripts/results/ct_full_lineage

# ============================================================
# 确定 embryo 列表
# ============================================================
if [ "$1" = "ALL" ]; then
    EMBRYO_LIST=$(ls -d "$RUN_DATA"/*/ 2>/dev/null | xargs -I {} basename {} | sort)
elif [ $# -gt 0 ]; then
    EMBRYO_LIST="$@"
else
    # 默认: 跑 2 个测试胚胎
    EMBRYO_LIST="170704plc1p1 200113plc1p2"
fi

echo "============================================"
echo "  CTransformer 完整 Pipeline"
echo "============================================"
echo "  数据源:    $RUN_DATA"
echo "  模型:      $CKPT"
echo "  分割输出:  $SEG_OUTPUT"
echo "  谱系输出:  $LINEAGE_OUTPUT"
echo "  胚胎列表:  $EMBRYO_LIST"
echo "============================================"

# 转为 Python 列表格式
EMBRYO_PYLIST=$(echo $EMBRYO_LIST | tr ' ' '\n' | sed "s/.*/'&'/" | paste -sd, -)
EMBRYO_PYLIST="[$EMBRYO_PYLIST]"
echo "  Python 列表: $EMBRYO_PYLIST"

# ============================================================
# Step 0: 安装依赖
# ============================================================
echo ""
echo "============================================"
echo "  Step 0: 检查 / 安装依赖"
echo "============================================"
python -c "import monai" 2>/dev/null || pip install monai==1.3.0
python -c "import torchmetrics" 2>/dev/null || pip install torchmetrics==0.11.4
python -c "import tiler" 2>/dev/null || pip install tiler
python -c "import treelib" 2>/dev/null || pip install treelib
python -c "import scipy" 2>/dev/null || pip install scipy
python -c "import nibabel" 2>/dev/null || pip install nibabel --no-deps
python -c "import skimage" 2>/dev/null || pip install scikit-image
python -c "import gtda" 2>/dev/null || pip install giotto-tda
python -c "import openpyxl" 2>/dev/null || pip install openpyxl  # CellFate.xls 读取
echo "依赖检查完成"

# ============================================================
# Step 1: 兼容性补丁
# ============================================================
echo ""
echo "============================================"
echo "  Step 1: 应用兼容性补丁"
echo "============================================"

# --- torch.load: 添加 weights_only=False ---
for f in $CT_ROOT/3_run_segmentation.py $CT_ROOT/run_segmentation.py; do
    if [ -f "$f" ] && grep -q "torch.load(" "$f" && ! grep -q "weights_only" "$f"; then
        sed -i 's/torch\.load(\(.*\))/torch.load(\1, weights_only=False)/g' "$f"
        echo "  [patched] torch.load: $(basename $f)"
    fi
done

# --- module. prefix: strip from state_dict ---
if [ -f "$CT_ROOT/3_run_segmentation.py" ] && ! grep -q "module\." "$CT_ROOT/3_run_segmentation.py" | grep -q "replace"; then
    # 只在还没 patch 时添加
    if ! grep -q 'k.replace("module."' "$CT_ROOT/3_run_segmentation.py"; then
        sed -i '/check_point = torch.load/a\    check_point["state_dict"] = {k.replace("module.", ""): v for k, v in check_point["state_dict"].items()}' "$CT_ROOT/3_run_segmentation.py"
        echo "  [patched] module. prefix strip"
    fi
fi

# --- num_workers: 改为 0 ---
if [ -f "$CT_ROOT/3_run_segmentation.py" ]; then
    sed -i 's/num_workers=1/num_workers=0/g' "$CT_ROOT/3_run_segmentation.py"
    echo "  [patched] num_workers=0"
fi

# --- np.float → float ---
for f in $CT_ROOT/segmentation_utils/ProcessLib.py $CT_ROOT/6_build_cell_shape_map.py $CT_ROOT/data_utils/augmentations.py; do
    if [ -f "$f" ] && grep -q 'np\.float)' "$f"; then
        sed -i 's/np\.float)/float)/g' "$f"
        echo "  [patched] np.float: $(basename $f)"
    fi
done

# --- collections.Sequence → collections.abc.Sequence ---
if [ -f "$CT_ROOT/data_utils/transforms.py" ] && grep -q 'collections\.Sequence' "$CT_ROOT/data_utils/transforms.py"; then
    sed -i 's/collections\.Sequence/collections.abc.Sequence/g' "$CT_ROOT/data_utils/transforms.py"
    echo "  [patched] collections.abc.Sequence"
fi

# --- scipy.ndimage.morphology (deprecated) ---
for f in $(grep -rl "scipy.ndimage.morphology" "$CT_ROOT" 2>/dev/null || true); do
    if [ -f "$f" ]; then
        sed -i 's/from scipy\.ndimage\.morphology import/from scipy.ndimage import/g' "$f"
        echo "  [patched] scipy.ndimage: $(basename $f)"
    fi
done

echo "补丁应用完成"

# ============================================================
# Step 2: 生成 Step 3 分割配置
# ============================================================
echo ""
echo "============================================"
echo "  Step 2: 生成分割配置 (Step 3)"
echo "============================================"

mkdir -p "$CT_ROOT/para_config"
cat > "$CT_ROOT/para_config/3_full_pipeline.yaml" << YAMLEOF
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

is_predict_memb: True
is_segment_cell: True

is_nuc_labelled: False
is_nuc_predicted: False
is_nuc_predicted_and_localmin: False

mem_edt_threshold: 9
YAMLEOF

echo "配置已写入: $CT_ROOT/para_config/3_full_pipeline.yaml"
echo "--- 配置内容 ---"
cat "$CT_ROOT/para_config/3_full_pipeline.yaml"
echo "--- 结束 ---"

# ============================================================
# Step 3: 运行 CTransformer 膜预测 + 细胞分割
# ============================================================
echo ""
echo "============================================"
echo "  Step 3: 运行 CTransformer 分割"
echo "============================================"

mkdir -p "$SEG_OUTPUT"
cd "$CT_ROOT"
python 3_run_segmentation.py -cfg 3_full_pipeline -gpu 0

echo ""
echo "分割完成! 检查输出:"
for embryo in $EMBRYO_LIST; do
    memb_count=$(find "$SEG_OUTPUT/$embryo/SegMemb" -name "*.nii.gz" 2>/dev/null | wc -l)
    cell_count=$(find "$SEG_OUTPUT/$embryo/SegCell" -name "*.nii.gz" 2>/dev/null | wc -l)
    echo "  $embryo: SegMemb=$memb_count, SegCell=$cell_count"
done

# ============================================================
# Step 4: 运行谱系映射 (Step 6)
# ============================================================
echo ""
echo "============================================"
echo "  Step 4: 运行谱系映射"
echo "============================================"

# 构建 --embryos 参数
EMBRYO_ARGS=""
for e in $EMBRYO_LIST; do
    EMBRYO_ARGS="$EMBRYO_ARGS $e"
done

cd "$PROJECT_ROOT"

FATE_ARG=""
if [ -f "$CELL_FATE" ]; then
    FATE_ARG="--cell-fate $CELL_FATE"
    echo "  CellFate: $CELL_FATE"
else
    echo "  [注意] CellFate.xls 不存在, 跳过凋亡细胞特殊处理"
fi

python scripts/run_lineage_mapping.py \
    --seg-root "$SEG_OUTPUT" \
    --annotated-root "$RUN_DATA" \
    --name-dict "$NAME_DICT" \
    $FATE_ARG \
    --output "$LINEAGE_OUTPUT" \
    --embryos $EMBRYO_ARGS

# ============================================================
# 完成
# ============================================================
echo ""
echo "============================================"
echo "  Pipeline 完成!"
echo "============================================"
echo "  分割输出:  $SEG_OUTPUT"
echo "  谱系输出:  $LINEAGE_OUTPUT"
echo ""
echo "  分割结果结构:"
echo "    {embryo}/SegMemb/{embryo}_{tp}_segMemb.nii.gz"
echo "    {embryo}/SegCell/{embryo}_{tp}_segCell.nii.gz"
echo ""
echo "  谱系结果结构:"
echo "    {embryo}/{embryo}_{tp}_segCell.nii.gz (重映射后)"
echo "    middle_materials/{embryo}/mapping/*.csv"
echo "    middle_materials/{embryo}/dividing/*.csv"
echo "    middle_materials/{embryo}/losing/*.csv"
