#!/bin/bash

python -m ocr_mllm_toy.eval.model_vqa_loader \
    --model-path ._1/checkpoints/qwen14b-finetune_all/checkpoint-8300 \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version/images \
    --answers-file ./playground/data/eval/MME/answers/ocr_mllm_toy-MME-14b.jsonl \
    --temperature 0 \
    --conv-mode mpt

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment ocr_mllm_toy-MME-14b

cd eval_tool

python calculation.py --results_dir answers/ocr_mllm_toy-MME-14b
