#!/bin/bash

# ================= 配置区域 =================
# 指定要使用的 GPU ID (例如: 0 1 2)
GPUS=(0 1 2)

# 数据集列表
# DATASETS=("diginetica" "retailRocket_DSAN" "Tmall" "Nowplaying")
DATASETS=("retailRocket_DSAN")

# 运行次数设置 (使用不同的 Seed 确保结果具有统计意义)
SEEDS=(42 43 44 45 46)

# 其他固定参数
EPOCH=15
BATCH_SIZE=100

# 日志存放目录
LOG_DIR="/data/UIO-SBR/compare/SPGL/logs"
mkdir -p $LOG_DIR
# ===========================================

# 创建一个临时管道文件作为令牌桶
FIFO_FILE="/tmp/$$.fifo"
mkfifo $FIFO_FILE
exec 3<>$FIFO_FILE
rm $FIFO_FILE

# 初始化令牌桶：将每张显卡的 ID 写入管道
for gpu_id in "${GPUS[@]}"; do
    echo $gpu_id >&3
done

# 捕捉 Ctrl+C 信号，终止所有子进程
trap 'exec 3>&-; kill $(jobs -p); exit' INT

echo "=========================================================="
echo "开始并行训练..."
echo "可用 GPU: ${GPUS[*]}"
echo "任务总数: $((${#DATASETS[@]} * ${#SEEDS[@]}))"
echo "日志目录: $LOG_DIR"
echo "=========================================================="

# 双层循环：遍历数据集和 Seed
for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        
        # 1. 从管道中读取一个可用的 GPU ID (如果管道为空，脚本会在此阻塞等待)
        read -u 3 gpu_id
        
        # 定义日志文件名
        log_file="${LOG_DIR}/${dataset}_seed${seed}.log"
        
        # 2. 在后台启动任务
        (
            echo "[$(date '+%H:%M:%S')] 启动任务: Dataset=$dataset | Seed=$seed | GPU=$gpu_id"
            
            # 运行 Python 脚本，并指定当前获取到的 GPU
            CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
                --dataset "$dataset" \
                --epoch $EPOCH \
                --batchSize $BATCH_SIZE \
                --seed $seed > "$log_file" 2>&1
            
            # 检查任务是否成功
            if [ $? -eq 0 ]; then
                echo "[$(date '+%H:%M:%S')] 任务完成: Dataset=$dataset | Seed=$seed | GPU=$gpu_id"
            else
                echo "[$(date '+%H:%M:%S')] 任务出错: Dataset=$dataset | Seed=$seed | 查看 $log_file"
            fi
            
            # 3. 任务结束后，将 GPU ID 归还给管道，供下一个任务使用
            echo $gpu_id >&3
        ) &
        
    done
done

# 等待所有后台任务完成
wait
echo "=========================================================="
echo "所有任务已完成！"