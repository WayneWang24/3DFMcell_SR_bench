#!/bin/bash
# =============================================================
# SR → 强度修复 → CTransformer 分割 + 谱系 对比 pipeline
#
# 在 Docker 容器内运行:
#   sudo docker start cellsr_eval_v2 && sudo docker exec -it cellsr_eval_v2 bash
#   cd /home/vm/workspace/3DFMcell_SR_bench
#   bash scripts/run_seg_comparison.sh [fastz|singleS] [xz|yz]
#
# 示例:
#   bash scripts/run_seg_comparison.sh fastz xz
#   bash scripts/run_seg_comparison.sh singleS yz
# =============================================================
set -e

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
CT_ROOT=$PROJECT_ROOT/CTransformer

# ============================================================
# 参数
# ============================================================
CONDITION=${1:-fastz}       # fastz 或 singleS
AXIS=${2:-xz}              # xz 或 yz
GPU_ID=${3:-0}             # GPU 编号，默认 0

# ============================================================
# 统一路径配置
# ============================================================
# singleS 目录名是 single_shot_volumes，fastz 是 fastz_volumes
if [ "$CONDITION" = "singleS" ]; then
    RAW_DIR=$PROJECT_ROOT/data/raw_cell_datasets/single_shot_volumes
else
    RAW_DIR=$PROJECT_ROOT/data/raw_cell_datasets/${CONDITION}_volumes
fi
SR_DIR=$PROJECT_ROOT/scripts/results/eval_outputs/sr_3d_${CONDITION}_${AXIS}
SR_FIXED_DIR=${SR_DIR}_fixed
CT_INPUT=$PROJECT_ROOT/data/ct_input/${CONDITION}_${AXIS}
SEG_OUTPUT=$PROJECT_ROOT/scripts/results/seg_comparison/${CONDITION}_${AXIS}
VIS_OUTPUT=$PROJECT_ROOT/scripts/results/seg_vis/${CONDITION}_${AXIS}
CKPT=$CT_ROOT/ckpts/sTUNETr_1_20240203/model_epoch_1000_edt6_sTUNETr.pth

cd "$PROJECT_ROOT"

echo "============================================"
echo "  SR Bench 分割对比 Pipeline"
echo "============================================"
echo "  PROJECT_ROOT: $PROJECT_ROOT"
echo "  条件:     $CONDITION"
echo "  切面:     $AXIS"
echo "  原始数据: $RAW_DIR"
echo "  SR 输出:  $SR_DIR"
echo "  SR 修复:  $SR_FIXED_DIR"
echo "  CT 输入:  $CT_INPUT"
echo "  分割输出: $SEG_OUTPUT"
echo "  可视化:   $VIS_OUTPUT"
echo "  模型:     $CKPT"
echo "============================================"

# ============================================================
# 路径验证
# ============================================================
echo ""
echo "[路径验证]"
if [ -d "$RAW_DIR" ]; then
    RAW_MEMB_COUNT=$(ls "$RAW_DIR"/memb_*.nii.gz 2>/dev/null | wc -l)
    RAW_NUC_COUNT=$(ls "$RAW_DIR"/nuc_*.nii.gz 2>/dev/null | wc -l)
    echo "  原始数据目录存在: $RAW_DIR"
    echo "    memb_*.nii.gz: $RAW_MEMB_COUNT 个"
    echo "    nuc_*.nii.gz:  $RAW_NUC_COUNT 个"
    if [ "$RAW_MEMB_COUNT" -eq 0 ]; then
        echo "  [错误] 原始数据目录没有 memb_*.nii.gz 文件!"
        echo "  目录内容:"
        ls -la "$RAW_DIR"/ | head -20
        exit 1
    fi
else
    echo "  [错误] 原始数据目录不存在: $RAW_DIR"
    echo "  请检查路径。data/raw_cell_datasets/ 下有:"
    ls -la "$PROJECT_ROOT/data/raw_cell_datasets/" 2>/dev/null || echo "    (目录不存在)"
    exit 1
fi

if [ -d "$SR_DIR" ]; then
    echo "  SR 目录存在: $SR_DIR"
else
    echo "  SR 目录不存在 (跳过 SR 处理): $SR_DIR"
fi

echo "  模型文件: $([ -f "$CKPT" ] && echo '存在' || echo '不存在')"

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
python -c "import matplotlib" 2>/dev/null || pip install matplotlib
echo "依赖检查完成"

# ============================================================
# Step 1: 兼容性补丁
# ============================================================
echo ""
echo "============================================"
echo "  Step 1: 应用兼容性补丁"
echo "============================================"

# torch.load: weights_only=False
for f in $CT_ROOT/3_run_segmentation.py $CT_ROOT/run_segmentation.py; do
    if [ -f "$f" ] && grep -q "torch.load(" "$f" && ! grep -q "weights_only" "$f"; then
        sed -i 's/torch\.load(\(.*\))/torch.load(\1, weights_only=False)/g' "$f"
        echo "  [patched] torch.load: $(basename $f)"
    fi
done

# module. prefix strip
if [ -f "$CT_ROOT/3_run_segmentation.py" ] && ! grep -q 'k.replace("module."' "$CT_ROOT/3_run_segmentation.py"; then
    sed -i '/check_point = torch.load/a\    check_point["state_dict"] = {k.replace("module.", ""): v for k, v in check_point["state_dict"].items()}' "$CT_ROOT/3_run_segmentation.py"
    echo "  [patched] module. prefix strip"
fi

# num_workers=0
if [ -f "$CT_ROOT/3_run_segmentation.py" ]; then
    sed -i 's/num_workers=1/num_workers=0/g' "$CT_ROOT/3_run_segmentation.py"
    echo "  [patched] num_workers=0"
fi

# np.float → float
for f in $CT_ROOT/segmentation_utils/ProcessLib.py $CT_ROOT/6_build_cell_shape_map.py $CT_ROOT/data_utils/augmentations.py; do
    if [ -f "$f" ] && grep -q 'np\.float)' "$f"; then
        sed -i 's/np\.float)/float)/g' "$f"
        echo "  [patched] np.float: $(basename $f)"
    fi
done

# collections.abc.Sequence
if [ -f "$CT_ROOT/data_utils/transforms.py" ] && grep -q 'collections\.Sequence' "$CT_ROOT/data_utils/transforms.py"; then
    sed -i 's/collections\.Sequence/collections.abc.Sequence/g' "$CT_ROOT/data_utils/transforms.py"
    echo "  [patched] collections.abc.Sequence"
fi

# scipy.ndimage.morphology deprecated
for f in $(grep -rl "scipy.ndimage.morphology" "$CT_ROOT" 2>/dev/null || true); do
    if [ -f "$f" ]; then
        sed -i 's/from scipy\.ndimage\.morphology import/from scipy.ndimage import/g' "$f"
        echo "  [patched] scipy.ndimage: $(basename $f)"
    fi
done

echo "补丁应用完成"

# ============================================================
# Step 2: 修复 SR 输出强度
# ============================================================
echo ""
echo "============================================"
echo "  Step 2: 修复 SR 输出强度 (percentile match)"
echo "============================================"

if [ -d "$SR_DIR" ]; then
    python "$PROJECT_ROOT/scripts/fix_sr_intensity.py" \
        --sr-dir "$SR_DIR" \
        --raw-dir "$RAW_DIR" \
        --output-dir "$SR_FIXED_DIR" \
        --method percentile
else
    echo "  [跳过] SR 目录不存在: $SR_DIR"
    SR_FIXED_DIR="$SR_DIR"
fi

# ============================================================
# Step 3: 准备 CTransformer 输入数据 (预 resize)
# ============================================================
echo ""
echo "============================================"
echo "  Step 3: 准备 CTransformer 输入数据"
echo "============================================"

python "$PROJECT_ROOT/scripts/prepare_ct_data.py" \
    --raw-dir "$RAW_DIR" \
    --sr-dir "$SR_FIXED_DIR" \
    --output-dir "$CT_INPUT" \
    --condition "$CONDITION" \
    --target-size 256 384 224

# 验证 prepare_ct_data 输出
echo ""
echo "[验证 CT 输入数据]"
if [ -d "$CT_INPUT" ]; then
    for d in "$CT_INPUT"/*/; do
        if [ -d "$d" ]; then
            embryo=$(basename "$d")
            memb_n=$(ls "$d/RawMemb"/*.nii.gz 2>/dev/null | wc -l)
            nuc_n=$(ls "$d/RawNuc"/*.nii.gz 2>/dev/null | wc -l)
            echo "  $embryo: RawMemb=$memb_n, RawNuc=$nuc_n"
        fi
    done
else
    echo "  [错误] CT 输入目录未创建: $CT_INPUT"
    exit 1
fi

# ============================================================
# Step 4: 收集 embryo 名称 + 生成配置
# ============================================================
echo ""
echo "============================================"
echo "  Step 4: 生成 CTransformer 配置"
echo "============================================"

EMBRYO_LIST=$(ls -d "$CT_INPUT"/*/ 2>/dev/null | xargs -I {} basename {} | sort)
EMBRYO_PYLIST=$(echo $EMBRYO_LIST | tr ' ' '\n' | sed "s/.*/'&'/" | paste -sd, -)
EMBRYO_PYLIST="[$EMBRYO_PYLIST]"
echo "  Embryo 列表: $EMBRYO_PYLIST"

CFG_NAME="3_SR_bench_${CONDITION}_${AXIS}"
mkdir -p "$CT_ROOT/para_config"
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

run_data_path: ${CT_INPUT}
output_data_path: ${SEG_OUTPUT}
run_transforms:
  Compose([
    NumpyType((np.float32, np.float32))
    ])

is_predict_memb: True
is_segment_cell: True

is_nuc_labelled: False
is_nuc_predicted: False
is_nuc_predicted_and_localmin: False

mem_edt_threshold: 9
YAMLEOF

echo "  配置: $CT_ROOT/para_config/${CFG_NAME}.yaml"
cat "$CT_ROOT/para_config/${CFG_NAME}.yaml"

# ============================================================
# Step 5: 运行 CTransformer 分割
# ============================================================
echo ""
echo "============================================"
echo "  Step 5: 运行 CTransformer 分割"
echo "============================================"

mkdir -p "$SEG_OUTPUT"
cd "$CT_ROOT"
python 3_run_segmentation.py -cfg "$CFG_NAME" -gpu "$GPU_ID"
cd "$PROJECT_ROOT"

echo ""
echo "分割完成! 检查输出:"
for embryo in $EMBRYO_LIST; do
    memb_count=$(find "$SEG_OUTPUT/$embryo/SegMemb" -name "*.nii.gz" 2>/dev/null | wc -l)
    cell_count=$(find "$SEG_OUTPUT/$embryo/SegCell" -name "*.nii.gz" 2>/dev/null | wc -l)
    echo "  $embryo: SegMemb=$memb_count, SegCell=$cell_count"
done

# ============================================================
# Step 6: 可视化
# ============================================================
echo ""
echo "============================================"
echo "  Step 6: 生成可视化 PNG"
echo "============================================"

python "$PROJECT_ROOT/scripts/visualize_seg.py" \
    --seg-root "$SEG_OUTPUT" \
    --data-root "$CT_INPUT" \
    --output "$VIS_OUTPUT"

# ============================================================
# 完成
# ============================================================
echo ""
echo "============================================"
echo "  Pipeline 完成!"
echo "============================================"
echo ""
echo "  输出路径总览:"
echo "    SR 修复:      $SR_FIXED_DIR"
echo "    CT 输入:      $CT_INPUT"
echo "    分割结果:     $SEG_OUTPUT"
echo "    可视化 PNG:   $VIS_OUTPUT"
echo ""
echo "  传到本地查看:"
echo "    tar -czf seg_${CONDITION}_${AXIS}.tar.gz \\"
echo "      $VIS_OUTPUT \\"
echo "      $SEG_OUTPUT"
