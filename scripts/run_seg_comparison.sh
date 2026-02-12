#!/bin/bash
# =============================================================
# SR → CTransformer 分割对比 pipeline
# 用法: bash scripts/run_seg_comparison.sh
# =============================================================
set -e

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
CT_ROOT=$PROJECT_ROOT/CTransformer
cd "$PROJECT_ROOT"

echo "============================================"
echo "  Step 1: 准备 CTransformer 输入数据"
echo "============================================"

python scripts/prepare_ct_data.py \
  --raw-dir data/raw_cell_datasets/fastz_volumes \
  --sr-dir scripts/results/eval_outputs/sr_3d_fastz_xz \
  --output-dir CTransformer/DataSource/RunningDataset \
  --condition fastz

echo ""
echo "============================================"
echo "  Step 2: 收集所有 embryo 名称"
echo "============================================"
EMBRYOS=$(ls -d CTransformer/DataSource/RunningDataset/*/ 2>/dev/null | xargs -I {} basename {} | tr '\n' "," | sed "s/,$//" | sed "s/,/','/g")
EMBRYOS="['${EMBRYOS}']"
echo "Embryo 列表: $EMBRYOS"

echo ""
echo "============================================"
echo "  Step 3: 生成 CTransformer 推断配置"
echo "============================================"

cat > "$CT_ROOT/para_config/3_SR_bench_running.yaml" << YAMLEOF
workflow_step: 3_RUN

net: SwinUNETR
net_params:
  img_size: (128,128,128)
  in_channels: 2
  feature_size: 48
  out_channels: 1

is_input_nuc_channel: True

running_embryo_names: ${EMBRYOS}

dataset_name: NiigzRunDataset
trained_model: ./ckpts/sTUNETr_1_20240203/model_epoch_1000_edt6_sTUNETr.pth
seed: 1024

run_data_path: ./DataSource/RunningDataset
output_data_path: ./OutputData/SR_bench
run_transforms:
  Compose([
    Resize((256, 384, 224)),
    NumpyType((np.float32, np.float32, np.float32, np.float32))
    ])

is_predict_memb: True
is_segment_cell: True

is_nuc_labelled: False
is_nuc_predicted: False
is_nuc_predicted_and_localmin: False

mem_edt_threshold: 9
YAMLEOF

echo "配置已写入: $CT_ROOT/para_config/3_SR_bench_running.yaml"
cat "$CT_ROOT/para_config/3_SR_bench_running.yaml"

echo ""
echo "============================================"
echo "  Step 4: 运行 CTransformer 分割推断"
echo "============================================"
cd "$CT_ROOT"
python 3_run_segmentation.py -cfg 3_SR_bench_running -gpu 0

echo ""
echo "============================================"
echo "  完成! 检查输出:"
echo "============================================"
find OutputData/SR_bench -name "*segCell*" -o -name "*segMemb*" | head -30
