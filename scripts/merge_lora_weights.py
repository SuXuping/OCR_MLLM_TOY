from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
import argparse

from ocr_mllm_toy.mm_utils import get_model_name_from_path
import os

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from ocr_mllm_toy.model import *
from ocr_mllm_toy.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from ocr_mllm_toy.model.language_model.baichuan2.configuration_baichuan import BaichuanConfig
from ocr_mllm_toy.model.language_model.baichuan2.tokenization_baichuan import BaichuanTokenizer
from ocr_mllm_toy.model.language_model.qwen14b.configuration_qwen import QWenConfig
from ocr_mllm_toy.model.language_model.qwen14b.modeling_qwen import QWenLMHeadModel
from ocr_mllm_toy.model.language_model.qwen14b.tokenization_qwen import QWenTokenizer

def get_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):
    kwargs = {"device_map": device_map}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if 'lora' in model_name.lower() and model_base is not None:
        if 'baichuan' in model_name.lower():
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = BaichuanTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading OCR_MLLM_TOY from base model...')
            model = OCR_MLLM_TOYBaichuanForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
        elif 'qwen' in model_name.lower():
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = QWenTokenizer.from_pretrained(model_base, use_fast=False)
            tokenizer.pad_token_id = tokenizer.eod_id
            print('Loading OCR_MLLM_TOY from base model...')
            model = OCR_MLLM_TOYQWenForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
        else:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading OCR_MLLM_TOY from base model...')
            model = OCR_MLLM_TOYLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

        print('Loading additional weights...')
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):  ###加载线性映射层
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')

        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if "qwen" in model_path:
            print(f'合并qwen')
            if any(k.startswith('model.transformer.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        else:
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')
    elif model_base is not None:   ###加载预训练时候，只有mm_projector的权重
        # this may be mm projector only
        print('Loading OCR_MLLM_TOY from base model...')
        if 'baichuan' in model_name.lower():
            print(f'正在加载：baichuan')
            tokenizer = BaichuanTokenizer.from_pretrained(model_base, use_fast=True)
            cfg_pretrained = BaichuanConfig.from_pretrained(model_path, trust_remote_code=True)
            model = OCR_MLLM_TOYBaiChuanForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained,**kwargs)
        elif 'qwen' in model_name.lower():
            print(f'正在加载：qwen')
            tokenizer = QWenTokenizer.from_pretrained(model_base, use_fast=True)
            tokenizer.pad_token_id = tokenizer.eod_id
            cfg_pretrained = QWenConfig.from_pretrained(model_path, trust_remote_code=True)
            model = OCR_MLLM_TOYQWenForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained,**kwargs)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            model = OCR_MLLM_TOYLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

        mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
        mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
        model.load_state_dict(mm_projector_weights, strict=False)



    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", False)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    # model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model

def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model = get_pretrained_model(args.model_path, args.model_base, model_name, device_map='cpu')
    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/cv/sxp/OCR_MLLM_TOY_new/checkpoints/OCR_MLLM_TOY-v1.5-qwen14b-finetune-2m-lora-2/checkpoint-6500")
    parser.add_argument("--model-base", type=str, default="./OCR_MLLM_TOY/qwen_pretrain")
    parser.add_argument("--save-model-path", type=str, default="/home/cv/sxp/OCR_MLLM_TOY_new/checkpoints/OCR_MLLM_TOY-v1.5-qwen14b-finetune-2m-lora-2/checkpoint-6500-merge")

    args = parser.parse_args()

    merge_lora(args)
