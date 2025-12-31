#!/bin/bash

# 定义基础命令
BASE_CMD="nohup python Main.py --data netflix --gpu 6"

# 定义--zero_shot的取值
ZERO_SHOT_VALUES=(1)

# 定义--user_aug_path的取值
USER_AUG_PATHS=("~data/netflix/user_profile/netflix_final_rlhf_testmaskv1_batch_4_step_2000/item_profile_embeddings.npy" "~data/netflix/user_profile/netflix_final_rlhf_testmaskv1_step_2000_origin/user_profile_embeddings.npy" "~data/netflix/user_profile/user_profile_embeddings.npy")
 # #
# 遍历所有组合并执行命令
for zs in "${ZERO_SHOT_VALUES[@]}"; do
    for path in "${USER_AUG_PATHS[@]}"; do
        # 构建完整命令
        safe_path=$(echo "$path" | tr '/' '_')
        FULL_CMD="$BASE_CMD --zero_shot $zs --user_aug_path $path > log/output_${zs}_${safe_path}.log 2>&1 &"
        
        # 打印即将执行的命令
        echo "执行命令: $FULL_CMD"
        
        # 执行命令
        eval $FULL_CMD &
        
        # 获取后台进程的PID
        PID=$!
        
        # 打印PID信息
        echo "进程PID: $PID"
        
        # 等待一段时间，避免同时启动过多进程
        sleep 1
    done
done

echo "所有任务已提交"