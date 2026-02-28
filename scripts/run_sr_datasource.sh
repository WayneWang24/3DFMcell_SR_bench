#!/bin/bash
# =============================================================
# 实验2: SR 增强 DataSource → 分割 → 谱系
#
# 流程:
#   Stage 1: SR 超分辨率 (GPU)
#   Stage 2: 强度修复 (CPU)
#   Stage 3: 准备 CTransformer 输入 (CPU, resize→256×384×224)
#   Stage 4: CTransformer 分割 (GPU+CPU)
#   Stage 5: 谱系映射 (CPU)
#
# 在 polyHPC Docker 容器内运行:
#   sudo docker start cellsr_eval && sudo docker exec -it cellsr_eval bash
#   cd /home/vm/workspace/3DFMcell_SR_bench
#   bash scripts/run_sr_datasource.sh edsr [GPU_ID]
#   bash scripts/run_sr_datasource.sh swinir 1
#   bash scripts/run_sr_datasource.sh esrgan 0
#
# 从某个 Stage 恢复:
#   bash scripts/run_sr_datasource.sh edsr 0 --from 3
#
# 只跑 SegCell 恢复:
#   bash scripts/run_sr_datasource.sh edsr 0 --resume-segcell
#
# 保留中间文件 (默认自动清理):
#   bash scripts/run_sr_datasource.sh edsr 0 --keep
# =============================================================
set -e

# ============================================================
# 参数解析
# ============================================================
MODEL=${1:?"用法: bash scripts/run_sr_datasource.sh <edsr|swinir|esrgan> [GPU_ID] [--from STAGE] [--resume-segcell]"}
GPU_ID=${2:-0}

FROM_STAGE=1
RESUME_SEGCELL=false
KEEP_INTERMEDIATES=false

shift 2 2>/dev/null || true
while [ $# -gt 0 ]; do
    case "$1" in
        --from)   FROM_STAGE=$2; shift 2 ;;
        --resume-segcell) RESUME_SEGCELL=true; shift ;;
        --keep)   KEEP_INTERMEDIATES=true; shift ;;
        *)        shift ;;
    esac
done

# 清理中间产物的函数
cleanup_intermediate() {
    local DIR="$1"
    local LABEL="$2"
    if [ "$KEEP_INTERMEDIATES" = true ]; then
        echo "  [保留] $LABEL: $DIR (--keep)"
        return
    fi
    if [ -d "$DIR" ]; then
        local SIZE=$(du -sh "$DIR" 2>/dev/null | cut -f1)
        echo "  [清理] $LABEL: $DIR ($SIZE)"
        rm -rf "$DIR"
        echo "  [已释放] $SIZE"
    fi
}

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
CT_ROOT=$PROJECT_ROOT/CTransformer

# ============================================================
# 路径配置
# ============================================================
DATASOURCE=/home/vm/workspace/DataSource
DATASOURCE_RD=$DATASOURCE/RunningDataset
NAME_DICT=$DATASOURCE/name_dictionary.csv
CELL_FATE=$DATASOURCE/CellFate.xls
CKPT_CT=$CT_ROOT/ckpts/sTUNETr_1_20240203/model_epoch_1000_edt6_sTUNETr.pth
CKPT_BASE=$PROJECT_ROOT/scripts/results/work_dirs
EXP2_ROOT=$PROJECT_ROOT/scripts/results/exp2_${MODEL}

# ============================================================
# 模型配置
# ============================================================
case $MODEL in
    edsr)
        CONFIG=$PROJECT_ROOT/configs/edsr_x4_cells_xy.py
        CKPT=$CKPT_BASE/edsr_x4_cells_fastz_xy_p1p99/best_PSNR_iter_20000.pth
        ;;
    swinir)
        CONFIG=$PROJECT_ROOT/configs/swinir_x4_cells_xy.py
        CKPT=$CKPT_BASE/swinir_x4_xy_fastz_200_highL_XY_p1p99/best_PSNR_iter_15000.pth
        ;;
    esrgan)
        CONFIG=$PROJECT_ROOT/configs/esrgan_x4_cells_xy_gan.py
        CKPT=$CKPT_BASE/esrgan_x4_fastz_xy_p1p99/best_PSNR_iter_25000.pth
        ;;
    *)
        echo "[错误] 未知模型: $MODEL (支持: edsr, swinir, esrgan)"
        exit 1
        ;;
esac

# ============================================================
# 发现胚胎列表
# ============================================================
EMBRYO_LIST=$(ls -d "$DATASOURCE_RD"/*/ 2>/dev/null | xargs -I {} basename {} | sort)
EMBRYO_COUNT=$(echo $EMBRYO_LIST | wc -w | tr -d ' ')

echo "============================================"
echo "  实验2: SR 增强 DataSource Pipeline"
echo "============================================"
echo "  模型:      $MODEL"
echo "  SR 配置:   $CONFIG"
echo "  SR 权重:   $CKPT"
echo "  CT 权重:   $CKPT_CT"
echo "  GPU:       $GPU_ID"
echo "  胚胎数:    $EMBRYO_COUNT"
echo "  输出根:    $EXP2_ROOT"
echo "  起始阶段:  Stage $FROM_STAGE"
echo "  保留中间:  $KEEP_INTERMEDIATES"
echo "============================================"

# 检查关键文件
if [ $FROM_STAGE -le 1 ] && [ ! -f "$CKPT" ]; then
    echo "[错误] SR 权重不存在: $CKPT"
    echo "  请在 polyHPC 上检查 scripts/results/work_dirs/ 下的实际文件名"
    exit 1
fi
if [ $FROM_STAGE -le 4 ] && [ ! -f "$CKPT_CT" ]; then
    echo "[错误] CTransformer 权重不存在: $CKPT_CT"
    exit 1
fi

mkdir -p "$EXP2_ROOT"

# ============================================================
# Step 0: 兼容性补丁
# ============================================================
echo ""
echo "============================================"
echo "  Step 0: 兼容性补丁"
echo "============================================"

# --- mmengine Adafactor 注册冲突 (SR 步骤需要) ---
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
    else:
        print(f'[ok] Adafactor already patched: {p}')
PYEOF

# --- CTransformer 补丁 ---
# torch.load: 添加 weights_only=False
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
fi

# np.float → float
for f in $CT_ROOT/segmentation_utils/ProcessLib.py $CT_ROOT/6_build_cell_shape_map.py $CT_ROOT/data_utils/augmentations.py; do
    if [ -f "$f" ] && grep -q 'np\.float)' "$f"; then
        sed -i 's/np\.float)/float)/g' "$f"
        echo "  [patched] np.float: $(basename $f)"
    fi
done

# collections.Sequence → collections.abc.Sequence
if [ -f "$CT_ROOT/data_utils/transforms.py" ] && grep -q 'collections\.Sequence' "$CT_ROOT/data_utils/transforms.py"; then
    sed -i 's/collections\.Sequence/collections.abc.Sequence/g' "$CT_ROOT/data_utils/transforms.py"
    echo "  [patched] collections.abc.Sequence"
fi

# scipy.ndimage.morphology (deprecated)
for f in $(grep -rl "scipy.ndimage.morphology" "$CT_ROOT" 2>/dev/null || true); do
    if [ -f "$f" ]; then
        sed -i 's/from scipy\.ndimage\.morphology import/from scipy.ndimage import/g' "$f"
        echo "  [patched] scipy.ndimage: $(basename $f)"
    fi
done

echo "补丁完成"

# ============================================================
# Stage 1: SR 超分辨率
# ============================================================
if [ $FROM_STAGE -le 1 ]; then
echo ""
echo "============================================"
echo "  Stage 1: SR 超分辨率 ($MODEL)"
echo "============================================"

for embryo in $EMBRYO_LIST; do
    SR_OUT=$EXP2_ROOT/sr_output/$embryo
    INPUT_DIR=$DATASOURCE_RD/$embryo/RawMemb

    # 跳过已完成
    if [ -d "$SR_OUT" ]; then
        DONE_COUNT=$(find "$SR_OUT" -maxdepth 1 -name "*.nii.gz" 2>/dev/null | wc -l | tr -d ' ')
        INPUT_COUNT=$(find "$INPUT_DIR" -maxdepth 1 -name "*rawMemb*.nii.gz" 2>/dev/null | wc -l | tr -d ' ')
        if [ "$DONE_COUNT" -ge "$INPUT_COUNT" ] && [ "$INPUT_COUNT" -gt 0 ]; then
            echo "  [跳过] $embryo: SR 已完成 ($DONE_COUNT/$INPUT_COUNT)"
            continue
        fi
    fi

    if [ ! -d "$INPUT_DIR" ]; then
        echo "  [跳过] $embryo: 无 RawMemb 目录"
        continue
    fi

    echo "  [处理] $embryo"
    mkdir -p "$SR_OUT"

    python "$PROJECT_ROOT/scripts/eval/sr_3d_nifti.py" \
        --input-dir "$INPUT_DIR" \
        --output-dir "$SR_OUT" \
        --config "$CONFIG" \
        --checkpoint "$CKPT" \
        --scale 4 --slice-axis xz \
        --device "cuda:$GPU_ID"

    # sr_3d_nifti.py 会创建模型名子目录，将文件提到上层
    if ls "$SR_OUT"/*/*.nii.gz 1>/dev/null 2>&1; then
        mv "$SR_OUT"/*/*.nii.gz "$SR_OUT/" 2>/dev/null || true
        rmdir "$SR_OUT"/*/ 2>/dev/null || true
    fi

    DONE=$(find "$SR_OUT" -maxdepth 1 -name "*.nii.gz" | wc -l | tr -d ' ')
    echo "  [完成] $embryo: $DONE 个文件"
done

echo "Stage 1 完成"
fi

# ============================================================
# Stage 2: 强度修复
# ============================================================
if [ $FROM_STAGE -le 2 ]; then
echo ""
echo "============================================"
echo "  Stage 2: 强度修复"
echo "============================================"

for embryo in $EMBRYO_LIST; do
    SR_DIR=$EXP2_ROOT/sr_output/$embryo
    FIXED_DIR=$EXP2_ROOT/sr_fixed/$embryo

    if [ ! -d "$SR_DIR" ]; then
        continue
    fi

    # 跳过已完成
    if [ -d "$FIXED_DIR" ]; then
        SR_COUNT=$(find "$SR_DIR" -maxdepth 1 -name "*.nii.gz" 2>/dev/null | wc -l | tr -d ' ')
        FIXED_COUNT=$(find "$FIXED_DIR" -maxdepth 1 -name "*.nii.gz" 2>/dev/null | wc -l | tr -d ' ')
        if [ "$FIXED_COUNT" -ge "$SR_COUNT" ] && [ "$SR_COUNT" -gt 0 ]; then
            echo "  [跳过] $embryo: 修复已完成 ($FIXED_COUNT/$SR_COUNT)"
            continue
        fi
    fi

    echo "  [处理] $embryo"

    # fix_sr_intensity.py: 无子目录时 relpath 基于 dirname(sr_dir)
    # 所以 output-dir 设为 sr_fixed/ (上层), 输出自动到 sr_fixed/{embryo}/
    python "$PROJECT_ROOT/scripts/fix_sr_intensity.py" \
        --sr-dir "$SR_DIR" \
        --raw-dir "$DATASOURCE_RD/$embryo/RawMemb" \
        --output-dir "$EXP2_ROOT/sr_fixed"
done

echo "Stage 2 完成"

# 清理 Stage 1 中间产物 (sr_output → sr_fixed 后不再需要)
cleanup_intermediate "$EXP2_ROOT/sr_output" "sr_output (Stage 1 中间产物)"

fi

# ============================================================
# Stage 3: 准备 CTransformer 输入
# ============================================================
if [ $FROM_STAGE -le 3 ]; then
echo ""
echo "============================================"
echo "  Stage 3: 准备 CTransformer 输入 (预 resize)"
echo "============================================"

python "$PROJECT_ROOT/scripts/prepare_ct_data_datasource.py" \
    --sr-dir "$EXP2_ROOT/sr_fixed" \
    --datasource "$DATASOURCE_RD" \
    --output-dir "$EXP2_ROOT/ct_input" \
    --target-size 256 384 224

echo "Stage 3 完成"

# 清理 Stage 2 中间产物 (sr_fixed → ct_input 后不再需要)
cleanup_intermediate "$EXP2_ROOT/sr_fixed" "sr_fixed (Stage 2 中间产物)"

fi

# ============================================================
# Stage 4: CTransformer 分割
# ============================================================
if [ $FROM_STAGE -le 4 ] || [ "$RESUME_SEGCELL" = true ]; then
echo ""
echo "============================================"
echo "  Stage 4: CTransformer 分割"
echo "============================================"

# 只跑 SegCell 恢复模式
if [ "$RESUME_SEGCELL" = true ]; then
    echo "  [恢复模式] 只补跑 SegCell"
    SEG_EMBRYOS=$(ls -d "$EXP2_ROOT/seg"/*/ 2>/dev/null | xargs -I {} basename {} | sort)
    if [ -n "$SEG_EMBRYOS" ]; then
        cd "$CT_ROOT"
        python "$PROJECT_ROOT/scripts/resume_segcell.py" \
            --seg-root "$EXP2_ROOT/seg" \
            --embryos $SEG_EMBRYOS \
            --timeout 1800 \
            --workers 8
        cd "$PROJECT_ROOT"
    else
        echo "  [跳过] 无分割输出需要恢复"
    fi
else
    # 正常分割流程

    # 找出需要分割的胚胎 (ct_input 中有数据但 seg 中没有 SegMemb)
    CT_EMBRYOS=""
    CT_EMBRYO_COUNT=0
    for embryo in $(ls -d "$EXP2_ROOT/ct_input"/*/ 2>/dev/null | xargs -I {} basename {} | sort); do
        MEMB_IN=$(find "$EXP2_ROOT/ct_input/$embryo/RawMemb" -name "*.nii.gz" 2>/dev/null | wc -l | tr -d ' ')
        MEMB_SEG=$(find "$EXP2_ROOT/seg/$embryo/SegMemb" -name "*.nii.gz" 2>/dev/null | wc -l | tr -d ' ')
        if [ "$MEMB_IN" -gt 0 ] && [ "$MEMB_SEG" -lt "$MEMB_IN" ]; then
            CT_EMBRYOS="$CT_EMBRYOS $embryo"
            CT_EMBRYO_COUNT=$((CT_EMBRYO_COUNT + 1))
        else
            if [ "$MEMB_IN" -gt 0 ]; then
                echo "  [跳过] $embryo: SegMemb 已完成 ($MEMB_SEG/$MEMB_IN)"
            fi
        fi
    done

    if [ $CT_EMBRYO_COUNT -eq 0 ]; then
        echo "  所有胚胎已完成分割"
    else
        echo "  待分割: $CT_EMBRYO_COUNT 个胚胎"

        # 转为 Python 列表
        EMBRYO_PYLIST=$(echo $CT_EMBRYOS | tr ' ' '\n' | sed "s/.*/'&'/" | paste -sd, -)
        EMBRYO_PYLIST="[$EMBRYO_PYLIST]"

        # 生成配置文件
        mkdir -p "$CT_ROOT/para_config"
        cat > "$CT_ROOT/para_config/3_exp2_${MODEL}.yaml" << YAMLEOF
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

run_data_path: ${EXP2_ROOT}/ct_input
output_data_path: ${EXP2_ROOT}/seg
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

        echo "  配置: $CT_ROOT/para_config/3_exp2_${MODEL}.yaml"
        echo "  胚胎: $EMBRYO_PYLIST"

        cd "$CT_ROOT"
        python 3_run_segmentation.py -cfg 3_exp2_${MODEL} -gpu $GPU_ID
        cd "$PROJECT_ROOT"
    fi

    # 检查 SegCell 是否有遗漏 (watershed 可能超时)
    echo ""
    echo "  检查分割输出..."
    NEED_RECOVERY=false
    for embryo in $(ls -d "$EXP2_ROOT/seg"/*/ 2>/dev/null | xargs -I {} basename {} | sort); do
        MEMB_N=$(find "$EXP2_ROOT/seg/$embryo/SegMemb" -name "*.nii.gz" 2>/dev/null | wc -l | tr -d ' ')
        CELL_N=$(find "$EXP2_ROOT/seg/$embryo/SegCell" -name "*.nii.gz" 2>/dev/null | wc -l | tr -d ' ')
        if [ "$CELL_N" -lt "$MEMB_N" ]; then
            echo "  [!] $embryo: SegMemb=$MEMB_N, SegCell=$CELL_N (缺 $((MEMB_N - CELL_N)))"
            NEED_RECOVERY=true
        else
            echo "  [ok] $embryo: SegMemb=$MEMB_N, SegCell=$CELL_N"
        fi
    done

    if [ "$NEED_RECOVERY" = true ]; then
        echo ""
        echo "  自动补跑缺失的 SegCell..."
        SEG_EMBRYOS=$(ls -d "$EXP2_ROOT/seg"/*/ 2>/dev/null | xargs -I {} basename {} | sort)
        cd "$CT_ROOT"
        python "$PROJECT_ROOT/scripts/resume_segcell.py" \
            --seg-root "$EXP2_ROOT/seg" \
            --embryos $SEG_EMBRYOS \
            --timeout 1800 \
            --workers 8
        cd "$PROJECT_ROOT"
    fi
fi

echo "Stage 4 完成"

# 清理 Stage 3 中间产物 (ct_input → seg 后不再需要)
cleanup_intermediate "$EXP2_ROOT/ct_input" "ct_input (Stage 3 中间产物)"

fi

# ============================================================
# Stage 5: 谱系映射
# ============================================================
if [ $FROM_STAGE -le 5 ] && [ "$RESUME_SEGCELL" = false ]; then
echo ""
echo "============================================"
echo "  Stage 5: 谱系映射"
echo "============================================"

FATE_ARG=""
if [ -f "$CELL_FATE" ]; then
    FATE_ARG="--cell-fate $CELL_FATE"
fi

python "$PROJECT_ROOT/scripts/run_lineage_mapping.py" \
    --seg-root "$EXP2_ROOT/seg" \
    --annotated-root "$DATASOURCE_RD" \
    --name-dict "$NAME_DICT" \
    $FATE_ARG \
    --output "$EXP2_ROOT/lineage" \
    --skip-done

echo "Stage 5 完成"
fi

# ============================================================
# 完成 — 汇总
# ============================================================
echo ""
echo "============================================"
echo "  Pipeline 完成! ($MODEL)"
echo "============================================"
echo "  输出根目录: $EXP2_ROOT"
echo ""
echo "  目录结构:"
if [ "$KEEP_INTERMEDIATES" = true ]; then
echo "    sr_output/{embryo}/*.nii.gz        — SR 超分输出"
echo "    sr_fixed/{embryo}/*.nii.gz         — 强度修复后"
echo "    ct_input/{embryo}/RawMemb+RawNuc/  — CTransformer 输入"
else
echo "    (sr_output, sr_fixed, ct_input 已自动清理)"
fi
echo "    seg/{embryo}/SegMemb+SegCell/      — 分割结果"
echo "    lineage/{embryo}/                  — 谱系映射结果"
echo ""

# 统计
for stage_dir in sr_output sr_fixed ct_input seg lineage; do
    DIR=$EXP2_ROOT/$stage_dir
    if [ -d "$DIR" ]; then
        N_EMBRYOS=$(ls -d "$DIR"/*/ 2>/dev/null | wc -l | tr -d ' ')
        echo "  $stage_dir: $N_EMBRYOS 个胚胎"
    fi
done
echo ""
echo "  验证命令:"
echo "    find $EXP2_ROOT/lineage -name 'mapping_dict*' | wc -l"
