#!/bin/bash
# =============================================================
# 批量运行所有条件×切面的 SR 分割对比
#
# 用法:
#   sudo docker start cellsr_eval_v2 && sudo docker exec -it cellsr_eval_v2 bash
#   cd /home/vm/workspace/3DFMcell_SR_bench
#   bash scripts/run_all_seg_comparison.sh
# =============================================================
set +e  # 单组失败不中止

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

CONDITIONS="fastz singleS"
AXES="xz yz"

TOTAL=0
SUCCESS=0
FAILED=""

echo "============================================"
echo "  批量 SR 分割对比"
echo "  条件: $CONDITIONS"
echo "  切面: $AXES"
echo "============================================"

for cond in $CONDITIONS; do
    for axis in $AXES; do
        TOTAL=$((TOTAL + 1))
        SR_DIR=$PROJECT_ROOT/scripts/results/eval_outputs/sr_3d_${cond}_${axis}

        echo ""
        echo "########################################"
        echo "  [$TOTAL] $cond × $axis"
        echo "########################################"

        if [ ! -d "$SR_DIR" ]; then
            echo "  [跳过] SR 目录不存在: $SR_DIR"
            FAILED="$FAILED  $cond-$axis (无SR输出)\n"
            continue
        fi

        bash scripts/run_seg_comparison.sh "$cond" "$axis"
        if [ $? -eq 0 ]; then
            SUCCESS=$((SUCCESS + 1))
            echo "  [$cond × $axis] 完成!"
        else
            FAILED="$FAILED  $cond-$axis (运行出错)\n"
            echo "  [$cond × $axis] 失败!"
        fi
    done
done

# ============================================================
# 汇总
# ============================================================
echo ""
echo "============================================"
echo "  全部完成!"
echo "============================================"
echo "  成功: $SUCCESS / $TOTAL"

if [ -n "$FAILED" ]; then
    echo "  失败:"
    echo -e "$FAILED"
fi

echo ""
echo "  输出目录:"
for cond in $CONDITIONS; do
    for axis in $AXES; do
        SEG="$PROJECT_ROOT/scripts/results/seg_comparison/${cond}_${axis}"
        VIS="$PROJECT_ROOT/scripts/results/seg_vis/${cond}_${axis}"
        if [ -d "$SEG" ]; then
            n=$(find "$SEG" -name "*segCell*" 2>/dev/null | wc -l)
            echo "    $cond × $axis: $n segCell files"
        fi
    done
done

echo ""
echo "  打包全部结果:"
echo "    tar -czf all_seg_results.tar.gz scripts/results/seg_comparison/ scripts/results/seg_vis/"
