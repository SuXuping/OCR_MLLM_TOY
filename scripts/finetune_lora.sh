# #!/bin/bash

# # deepspeed llava/train/train_mem.py \  mm_projector映射层以及llm的线性层的lora训练
# deepspeed --include=localhost:2,3 --master_port 9913 llava/train/train_mem.py \
#     --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
#     --deepspeed ./scripts/zero3.json \
#     --model_name_or_path ./llava/baichuan_pretrain \
#     --version v1 \
#     --data_path /home/cv/sxp/LLaVA/datasets/vqa_llava_1_5_train.json \
#     --image_folder "" \
#     --vision_tower ./llava/clip-vit-large-patch14-336 \
#     --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-baichuan-13b-5m-pretrain/checkpoint-6500/mm_projector.bin \
#     --mm_projector_type mlp2x_gelu \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio pad \
#     --group_by_modality_length True \
#     --bf16 True \
#     --output_dir ./checkpoints/llava-v1.5-baichuan-13b-600k-lora-finetune \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 8 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 500 \
#     --save_total_limit 4 \
#     --learning_rate 2e-4 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb


#!/bin/bash
# resume
# deepspeed llava/train/train_mem.py \  mm_projector映射层以及llm的线性层的lora训练
deepspeed --include=localhost:2,3 --master_port 9913 llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./llava/baichuan_pretrain \
    --version v1 \
    --data_path /home/cv/sxp/LLaVA/datasets/vqa_llava_1_5_train.json \
    --image_folder "" \
    --vision_tower ./llava/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-baichuan-13b-600k-lora-finetune2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 4 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
