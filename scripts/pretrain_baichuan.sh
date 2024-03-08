#!/bin/bash
# deepspeed --include=localhost:1 pretrain_image_caption.py
# --image_folder /home/cv/sxp/LLaVA_1_0/dataset/cc3m_595k_images \
# --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-qwen14b-pretrain-5m-448/checkpoint-2500/pytorch_model.bin \

deepspeed --include=localhost:2 --master_port 9911 ocr_mllm_toy/train/train_mem.py \
    --deepspeed ./zeros/zero3.json \
    --model_name_or_path ./ocr_mllm_toy/pretrain_weight/baichuan_pretrain \
    --train_mode pretrain \
    --version plain \
    --data_path ./datasets/max_len_test.json \
    --image_folder "" \
    --vision_tower ./ocr_mllm_toy/pretrain_weight/qwen_vit_448 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-baichuan-pretrain-maxlen-test-zero3 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1 \
    --save_total_limit 4 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
