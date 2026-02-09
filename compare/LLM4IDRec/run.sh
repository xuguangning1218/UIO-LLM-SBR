
DATASET=diginetica
ROOT_PATH=/data/UIO-LLM-SBR/compare/LLM4IDRec
CUDA_ID=0
BASED_LLM=meta-llama/Meta-Llama-3-8B
IDICATOR=LLM4IDRec
AUG_SIZE=1
AUG_RATIO=0.1

cd ${ROOT_PATH}

python ${ROOT_PATH}/data_process_without_userid.py --dataset $DATASET --user_id_aug_size $AUG_SIZE --target_aug_ratio $AUG_RATIO &&

CUDA_VISIBLE_DEVICES=$CUDA_ID python ${ROOT_PATH}/tokenize_dataset_rows.py \
    --model_checkpoint $BASED_LLM \
    --dataset $DATASET \
    --idicator $IDICATOR \
    --max_seq_length 2048 \
    --user_id_aug_size $AUG_SIZE \
    --target_aug_ratio $AUG_RATIO \
    --skip_overlength False &&

CUDA_VISIBLE_DEVICES=$CUDA_ID python ${ROOT_PATH}/lora_tuning.py \
    --dataset $DATASET \
    --idicator $IDICATOR \
    --model_path $BASED_LLM \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --max_steps 400 \
    --save_steps 50 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --fp16 True \
    --remove_unused_columns false \
    --user_id_aug_size $AUG_SIZE \
    --target_aug_ratio $AUG_RATIO \
    --logging_steps 10 &&

CUDA_VISIBLE_DEVICES=$CUDA_ID python ${ROOT_PATH}/vllm_predict.py\
    --model_path $BASED_LLM \
    --dataset $DATASET \
    --user_id_aug_size $AUG_SIZE \
    --target_aug_ratio $AUG_RATIO \
    --idicator $IDICATOR &&

python ${ROOT_PATH}/data_augmentation_by_llm.py --dataset $DATASET --idicator $IDICATOR --user_id_aug_size $AUG_SIZE --target_aug_ratio $AUG_RATIO &&

cd ../.. &&

python main.py --dataset $DATASET --augmentation $IDICATOR
