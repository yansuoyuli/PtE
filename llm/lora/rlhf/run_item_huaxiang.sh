#!/bin/bash

# 确保脚本遇到错误时停止执行
set -e

# 定义颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # 无颜色

# 定义日志目录
LOG_DIR="logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo -e "${GREEN}===== 开始LLM训练和推理流程 ====${NC}"
echo "日志将保存在: $LOG_DIR"

# 函数: 执行带nohup的命令并等待完成
run_step() {
    local command=$1
    local log_file=$2
    local step_name=$3
    
    echo -e "${YELLOW}===== 开始执行: $step_name ====${NC}"
    
    # 使用nohup运行命令，并捕获PID
    nohup $command > "$LOG_DIR/$log_file.log" 2> "$LOG_DIR/$log_file.log" &
    local pid=$!
    echo "进程ID: $pid"
    echo "$pid" > "$LOG_DIR/$log_file.pid"
    
    # 等待命令完成
    wait $pid
    local exit_status=$?
    
    if [ $exit_status -eq 0 ]; then
        echo -e "${GREEN}$step_name 已成功完成 ====${NC}"
        return 0
    else
        echo -e "${RED}$step_name 失败，退出状态: $exit_status ====${NC}"
        echo "错误日志: $LOG_DIR/$log_file.log"
        return $exit_status
    fi
}

# 第一部分：奖励模型训练
# run_step "python reward_modeling.py" "reward_modeling" "奖励模型训练" || exit 1

# 第二部分：近端策略优化
# run_step "python rl_training.py" "rl_training" "近端策略优化" || exit 1

# 第三部分：协作指令微调(项目侧)
# run_step "python ../sft_base_item.py" "sft_base_item" "协作指令微调(项目侧)" || exit 1

# 第四部分：用户/项目画像生成
# 用户画像生成(仅知识蒸馏)
# run_step "python ../inference_base.py" "inference_base" "用户画像生成(仅知识蒸馏)" || exit 1

# 用户画像生成(含协作指令调优和强化学习增强)
# run_step "python ../inference_base_mask.py" "inference_base_mask" "用户画像生成(含增强)" || exit 1

# 项目画像生成
run_step "python ../inference_base_item.py" "inference_base_item" "项目画像生成" || exit 1

echo -e "${GREEN}===== LLM训练和推理流程全部完成! ====${NC}"
echo "日志保存在: $LOG_DIR"    