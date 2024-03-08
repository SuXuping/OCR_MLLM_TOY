from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
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
from .multimodal_encoder.utils import train_transform,test_transform

def load_pretrained_model(model_path, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):
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

    if 'baichuan' in model_name.lower():
        tokenizer = BaichuanTokenizer.from_pretrained(model_path, use_fast=True)
        model = OCR_MLLM_TOYBaichuanForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    elif 'qwen' in model_name.lower():
        tokenizer = QWenTokenizer.from_pretrained(model_path, use_fast=True)
        tokenizer.pad_token_id = tokenizer.eod_id
        model = OCR_MLLM_TOYQWenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = OCR_MLLM_TOYLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    
    # model_args = AutoConfig.from_pretrained(model_path)
    # model_args.pretrain_mm_mlp_adapter = os.path.join(model_path,"mm_projector.bin")
    # model.get_model().initialize_vision_weights(model_args)

    # mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    # mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", False)
    # if mm_use_im_patch_token:
    #     tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    # if mm_use_im_start_end:
    #     tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    # # model.resize_token_embeddings(len(tokenizer))

    # ###将除了llm之外的部分加载
    # vision_tower = model.get_vision_tower()
    # vision_tower.to(device=device, dtype=torch.float16)
    # image_processor = vision_tower.image_processor

    # vision_tower_high = model.get_vision_tower_high()
    # vision_tower_high.to(device=device, dtype=torch.float16)
    # image_processor_high = vision_tower_high.image_processor_high
        
    # if hasattr(model.config, "max_sequence_length"):
    #     context_len = model.config.max_sequence_length
    # else:
    #     context_len = 2048

    # model.get_model().mm_projector.to(device=device, dtype=torch.float16)
    # model.get_model().mm_projector_high.to(device=device, dtype=torch.float16)
    
    # return tokenizer, model, image_processor, image_processor_high, context_len

    model_args = QWenConfig.from_pretrained(model_path)
    model_args.pretrain_mm_mlp_adapter = os.path.join(model_path,"mm_projector.bin")
    model.get_model().initialize_vision_weights(model_args)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", False)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    # model.resize_token_embeddings(len(tokenizer))

    ###将除了llm之外的部分加载
    # vision_tower = model.get_vision_tower()
    model.get_vision_tower().to(device=device, dtype=torch.float16)
    # model.get_vision_tower().to(device="cpu", dtype=torch.float32)
    image_processor = model.get_vision_tower().image_processor

    # vision_tower_high = model.get_vision_tower_high()
    model.get_vision_tower_high().to(device=device, dtype=torch.float16)
    # model.get_vision_tower_high().to(device="cpu", dtype=torch.float32)
    image_processor_high = model.get_vision_tower_high().image_processor_high

    model.get_model().mm_projector.to(device=device, dtype=torch.float16)
    model.get_model().mm_projector_high.to(device=device, dtype=torch.float16)
    
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    
    return tokenizer, model, image_processor, image_processor_high, context_len