#!/bin/bash

python -m ocr_mllm_toy.eval.model_vqa_loader \
    --model-path ._1/checkpoints/qwen14b-finetune_all/checkpoint-8300 \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /opt/MM_LLM/OCR-DATA/VQA/TextVQA/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/ocr_mllm_toy-14b.jsonl \
    --temperature 0 \
    --conv-mode mpt

python -m ocr_mllm_toy.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llava-v1.5-13b.jsonl
