#!/bin/bash
set -x

wandb login

export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=8
export NNODES=1
export MASTER_PORT=29517
export CPUS_PER_TASK=32
export QUOTA=reserved

# export WANDB_MODE=disabled
export WANDB_PROJECT=llava-hound
export WANDB_NAME=dpo
export model_name_or_path=/mnt/hwfile/mllm/liuziyu/finetune_LLaVa/merge_lora_RLHF_llava_mix_textvqa_20k_coco_25k
export data_path=/mnt/petrelfs/liuziyu/RLHF/make_data/data_randomsample_randompic_45k/dpo_llava_format_textvqa_coco_12k_gaussion_noise_25_temperature_0_5.json
export video_dir=/
export image_dir=/mnt/hwfile/mllm/chenlin/llava/data/
export output_dir=/mnt/hwfile/mllm/liuziyu/finetune_LLaVa/dpo_llava_textvqa_coco_12k_gaussion_noise_25_temperature_0_5
export lr=5e-5
export cache_dir=/mnt/hwfile/mllm/liuziyu/video_data/cache


SRUN_ARGS=${SRUN_ARGS:-""}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p mllm \
    --nodes=$NNODES \
    --ntasks-per-node=1 \
    --gres=gpu:$GPUS_PER_NODE \
    --cpus-per-task=$CPUS_PER_TASK \
    --kill-on-bad-exit=1 \
    --quotatype=${QUOTA} \
    --time=1-10:00:00 \
    ${SRUN_ARGS} \
    bash -c 'torchrun --nnodes $NNODES --nproc_per_node $GPUS_PER_NODE --node_rank $SLURM_NODEID --master_addr $(scontrol show hostname $SLURM_NODELIST | head -n1) --master_port ${MASTER_PORT} dpo_scripts/run_dpo.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed config/zero2.json \
    --model_name_or_path ${model_name_or_path} \
    --dpo_alpha 1.0 --beta 0.1 --gamma 0 \
    --version v1 \
    --data_path ${data_path} \
    --video_folder ${video_dir} \
    --image_folder ${image_dir} \
    --X "Image" --training_modal 'image' \
    --image_tower /mnt/hwfile/mllm/liuziyu/CLIP_models/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_x_start_end False \
    --mm_use_x_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_only_model True \
    --save_total_limit 11 \
    --learning_rate ${lr} --freeze_mm_mlp_adapter True \
    --weight_decay 0. --warmup_ratio 0.1 \
    --lr_scheduler_type "linear" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --cache_dir ${cache_dir} \
    --report_to wandb '