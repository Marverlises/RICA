#!/bin/bash

# RICA Training Script
# Usage: bash run_rica.sh [dataset_name] [gpu_id] [batch_size]

DATASET_NAME=${1:-"CUHK-PEDES"}
GPU_ID=${2:-"0"}
BATCH_SIZE=${3:-"64"}

echo "Starting RICA training..."
echo "Dataset: $DATASET_NAME"
echo "GPU: $GPU_ID"
echo "Batch Size: $BATCH_SIZE"

CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
    --dataset_name $DATASET_NAME \
    --batch_size $BATCH_SIZE \
    --name rica_${DATASET_NAME} \
    --output_dir ./logs \
    --loss_names "id+gl" \
    --delta 4.0 \
    --sigma 0.08 \
    --lr 1e-5 \
    --num_epoch 60 \
    --img_aug \
    --MLM \
    --pretrain_choice "ViT-B/16" \
    --temperature 0.02 \
    --cmt_depth 4 \
    --mask_token_ratio 0.15 \
    --lr_factor 5.0 \
    --mlm_loss_weight 1.0 \
    --id_loss_weight 1.0 \
    --measure "dot" \
    --max_violation \
    --alignment_mode "MrSw" \
    --cap_dim 512 \
    --dropout 0.1 \
    --aggregation_type "first" \
    --img_dim 512 \
    --img_size "(384, 128)" \
    --stride_size 16 \
    --text_length 77 \
    --vocab_size 49408 \
    --optimizer "Adam" \
    --bias_lr_factor 2.0 \
    --momentum 0.9 \
    --weight_decay 4e-5 \
    --weight_decay_bias 0.0 \
    --alpha 0.9 \
    --beta 0.999 \
    --milestones "(20, 50)" \
    --gamma 0.1 \
    --warmup_factor 0.1 \
    --warmup_epochs 5 \
    --warmup_method "linear" \
    --lrscheduler "cosine" \
    --target_lr 0 \
    --power 0.9 \
    --sampler "random" \
    --num_instance 4 \
    --root_dir "./data/" \
    --test_batch_size 512 \
    --num_workers 8 \
    --log_period 100 \
    --eval_period 1 \
    --val_dataset "test" \
    --resume \
    --resume_ckpt_file ""

echo "Training completed!"