from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
import argparse
import torch

from ocr_mllm_toy.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from ocr_mllm_toy.conversation import conv_templates, SeparatorStyle
from ocr_mllm_toy.model.builder import load_pretrained_model
from ocr_mllm_toy.utils import disable_torch_init
from ocr_mllm_toy.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer,TextIteratorStreamer
from threading import Thread

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def parse_args():
    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument("--model-path", type=str, default="./checkpoints/qwen14b-finetune_all/checkpoint-8300")  ###也可以是model-base设置为None，然后model-path设置为lora merge之后的权重
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--conv-mode", type=str, default="mpt")
    parser.add_argument("--load-8bit", type=bool, default=True)
    parser.add_argument("--load-4bit", type=bool, default=False)
    args = parser.parse_args()
    return args

class Run_Inference():
    def __init__(self,args) -> None:
        # Model
        disable_torch_init()
        model_name = get_model_name_from_path(args.model_path)
        self.tokenizer, self.model, self.image_processor, self.image_processor_high, self.context_len = load_pretrained_model(args.model_path, model_name, args.load_8bit, args.load_4bit, device=args.device)
        self.conv = conv_templates[args.conv_mode].copy()

    def chat(self,**kwargs):
        torch.cuda.empty_cache()
        if kwargs['upload_iamge'] is True:  ###上传图片，历史清空
            assert kwargs['image_path'] is not None,"upload_iamge为True时候，image_path不能为None"
            image = load_image(kwargs['image_path'])
            # Similar operation in model_worker.py
            # image_tensor = process_images([image], self.image_processor, self.model.config)
            image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
            image_tensor_high = self.image_processor_high(image).unsqueeze(0)
            if type(image_tensor) is list:
                image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
                image_tensor_high = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor_high]
            else:
                image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
                image_tensor_high = image_tensor_high.to(self.model.device, dtype=torch.float16)

            query_text = kwargs['query_text']
            kwargs['conv_history'] = None
            conv = self.conv
            conv.messages = []
            if kwargs['conv_history'] is None: ###首轮对话
                query_text = DEFAULT_IMAGE_TOKEN + '\n' + query_text
                conv.append_message(conv.roles[0], query_text)
            else:
                # later messages
                conv = kwargs['conv_history']
                conv.append_message(conv.roles[0], query_text)
        elif kwargs['upload_iamge'] is False and kwargs['image_path'] is not None:  ##多模态对话
            image = load_image(kwargs['image_path'])
            # Similar operation in model_worker.py
            # image_tensor = process_images([image], self.image_processor, self.model.config)
            image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
            image_tensor_high = self.image_processor_high(image).unsqueeze(0)
            if type(image_tensor) is list:
                image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
                image_tensor_high = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor_high]
            else:
                image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
                image_tensor_high = image_tensor_high.to(self.model.device, dtype=torch.float16)

            query_text = kwargs['query_text']
            if kwargs['conv_history'] is None: ###首轮对话
                query_text = DEFAULT_IMAGE_TOKEN + '\n' + query_text
                conv = self.conv
                conv.messages = []
                conv.append_message(conv.roles[0], query_text)
            elif DEFAULT_IMAGE_TOKEN not in kwargs['conv_history'].get_prompt(): ##历史信息是纯文本对话
                # later messages
                conv = kwargs['conv_history']
                query_text = DEFAULT_IMAGE_TOKEN + '\n' + query_text
                conv.append_message(conv.roles[0], query_text)               
            else: ##多轮对话
                conv = kwargs['conv_history']
                conv.append_message(conv.roles[0], query_text)

        elif kwargs['upload_iamge'] is False and kwargs['image_path'] is None:   ###纯文本对话
            image_tensor = None
            image_tensor_high = None
            query_text = kwargs['query_text']
            if kwargs['conv_history'] is None: ###首轮对话
                query_text = query_text
                conv = self.conv
                conv.messages = []
                conv.append_message(conv.roles[0], query_text)
            else:
                # later messages
                conv = kwargs['conv_history']
                conv.append_message(conv.roles[0], query_text)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # print(f'当前的promt是：： {prompt}')
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)  ###input_ids中加入IMAGE_TOKEN_INDEX -200
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        # keywords = ["<|im_start|>", "<|im_end|>"]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        # streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        if kwargs['use_streamer']:
            with torch.inference_mode():
                generation_kwargs = dict(
                    input_ids=input_ids,
                    images=image_tensor,
                    images_high=image_tensor_high,
                    do_sample=True if kwargs['temperature'] > 0 else False,
                    # do_sample=False,
                    temperature=kwargs['temperature'],
                    max_new_tokens=kwargs['max_new_tokens'],
                    top_k=kwargs['top_k'],
                    top_p=kwargs['top_p'],
                    repetition_penalty=1.1,  ###input_id中的image token为-200，但是再repetition_penalty中（RepetitionPenaltyLogitsProcessor）input_id不能为负数
                    num_beams=1,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])
                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()
                # streamer_copy = copy.deepcopy(streamer)
                # generated_text = ""
                # for new_text in streamer_copy:
                #     generated_text += new_text
                # conv.messages[-1][-1] = generated_text
                return streamer,conv
        else:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    images_high=image_tensor_high,
                    # do_sample=False,
                    do_sample=True if kwargs['temperature'] > 0 else False,
                    temperature=kwargs['temperature'],
                    max_new_tokens=kwargs['max_new_tokens'],
                    top_k=kwargs['top_k'],
                    top_p=kwargs['top_p'],
                    repetition_penalty=1.1,  ###input_id中的image token为-200，但是再repetition_penalty中（RepetitionPenaltyLogitsProcessor）input_id不能为负数
                    num_beams=1,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])
            outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            conv.messages[-1][-1] = outputs
            print(outputs)
            return outputs,conv

if __name__ == "__main__":
    args = parse_args()
    model = Run_Inference(args)
    ##纯文本
    outputs,conv_history = model.chat(upload_iamge=False,image_path=None,query_text = "你是谁？", use_streamer=True,conv_history=None,temperature=0.3,max_new_tokens=2048,top_k=0,top_p=0.8)
    generated_text = ""
    for new_text in outputs:
        generated_text += new_text
    conv_history.messages[-1][-1] = generated_text
    print(generated_text)
    outputs,conv_history = model.chat(upload_iamge=False,image_path=None,query_text = "介绍一下小i机器人", use_streamer=True,conv_history=None,temperature=0.3,max_new_tokens=2048,top_k=0,top_p=0.8)
    generated_text = ""
    for new_text in outputs:
        generated_text += new_text
    conv_history.messages[-1][-1] = generated_text
    print(generated_text)
    ##新的多模态对话
    outputs,conv_history = model.chat(upload_iamge=True,image_path="./test_images/mulu.png",query_text = "图中有哪些文字？", use_streamer=True,conv_history=None,temperature=0.3,max_new_tokens=2048,top_k=0,top_p=0.8)
    generated_text = ""
    for new_text in outputs:
        generated_text += new_text
    conv_history.messages[-1][-1] = generated_text
    print(generated_text)
    ##历史对话多模态
    outputs,conv_history = model.chat(upload_iamge=False,image_path="./test_images/mulu.png",query_text = "画矩形在第几页？", use_streamer=True,conv_history=conv_history,temperature=0.3,max_new_tokens=2048,top_k=0,top_p=0.8)
    generated_text = ""
    for new_text in outputs:
        generated_text += new_text
    conv_history.messages[-1][-1] = generated_text
    print(generated_text)