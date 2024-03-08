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
from transformers import TextStreamer
from transformers import GenerationConfig

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, image_processor_high, context_len = load_pretrained_model(args.model_path, model_name, args.load_8bit, args.load_4bit, device=args.device)
    # generation_config = GenerationConfig.from_pretrained(args.model_path, pad_token_id=tokenizer.pad_token_id)
    # print(f'generation_config:::{generation_config}')

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    image = load_image(args.image_file)
    # Similar operation in model_worker.py
    # image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
    image_tensor_high = image_processor_high(image).unsqueeze(0)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        image_tensor_high = [image.to(model.device, dtype=torch.float16) for image in image_tensor_high]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        image_tensor_high = image_tensor_high.to(model.device, dtype=torch.float16)

    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")
        torch.cuda.empty_cache()
        print(f'使用torch.cuda.empty_cache()')
        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print(f'当前的promt是：： {prompt}')
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)  ###input_ids中加入IMAGE_TOKEN_INDEX -200
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        # keywords = ["<|endoftext|>"]
        # keywords = [[tokenizer.im_end_id], [tokenizer.im_start_id]]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        if -200 in input_ids:
            input_ids = input_ids[:,-1400:]
        else:
            input_ids = input_ids[:,-2000:]    
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                images_high=image_tensor_high,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                top_k=0,
                top_p=0.8,
                repetition_penalty=1.1,  ###input_id中的image token为-200，但是再repetition_penalty中（RepetitionPenaltyLogitsProcessor）input_id不能为负数
                num_beams=1,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs
        #print(outputs)
        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./checkpoints/qwen14b-finetune_all/checkpoint-8300")  ###也可以是model-base设置为None，然后model-path设置为lora merge之后的权重
    parser.add_argument("--model-base", type=str, default=None)  ###或者是model-base设置为LLM的权重路径，然后model-path设置为lora 的权重，但是如果是使用int8或4就只能加载lora merge之后的权重了
    parser.add_argument("--image-file", type=str, default="./test_images/mulu.png")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--conv-mode", type=str, default="mpt")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--load-8bit", type=bool, default=True)
    parser.add_argument("--load-4bit", type=bool, default=False)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
