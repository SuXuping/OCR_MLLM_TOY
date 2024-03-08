#!/bin/bash
#     --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-qwen14b-pretrain-5m-448-2/mm_projector.bin \
# deepspeed llava/train/train_mem.py \  mm_projector映射层以及llm的线性层的lora训练
deepspeed --include=localhost:2 --master_port 9913 ocr_mllm_toy/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./zeros/zero3.json \
    --model_name_or_path ./checkpoints/llava-v1.5-qwen14b-pretrain-1m-ocr-448-1024-zero3 \
    --train_mode finetune_lora \
    --version mpt \
    --data_path ./datasets/max_len_test.json \
    --image_folder "" \
    --vision_tower ./ocr_mllm_toy/pretrain_weight/qwen_vit_448 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-qwen14b-finetune-test-zero3 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5 \
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


# #!/bin/bash
# # resume
# # deepspeed llava/train/train_mem.py \  mm_projector映射层以及llm的线性层的lora训练
# deepspeed --include=localhost:1,2,3 --master_port 9913 llava/train/train_mem.py \
#     --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
#     --deepspeed ./scripts/zero2.json \
#     --model_name_or_path ./llava/qwen_pretrain \
#     --version mpt \
#     --data_path /home/cv/sxp/LLaVA/datasets/vqa_train.json \
#     --image_folder "" \
#     --vision_tower ./llava/qwen_clip \
#     --mm_projector_type mlp2x_gelu \
#     --mm_vision_select_layer -1 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio pad \
#     --group_by_modality_length True \
#     --bf16 True \
#     --output_dir ./checkpoints/llava-v1.5-qwen14b-finetune-2m-448 \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 16 \
#     --gradient_accumulation_steps 16 \
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
