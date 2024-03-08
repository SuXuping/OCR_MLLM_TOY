#!/bin/bash

SPLIT="mmbench_dev_20230712"

python -m ocr_mllm_toy.eval.model_vqa_mmbench \
    --model-path ./checkpoints/qwen14b-finetune_all/checkpoint-8300 \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/ocr_mllm_toy-MMbench-14b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode mpt

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment ocr_mllm_toy-MMbench-14b
