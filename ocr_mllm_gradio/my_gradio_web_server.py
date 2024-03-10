from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
print(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
import datetime
import json
import gradio as gr
import requests
import hashlib
import time
import uuid
import torch
from transformers import TextIteratorStreamer
from threading import Thread
import argparse
import os

from ocr_mllm_toy.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from ocr_mllm_toy.conversation import conv_templates, SeparatorStyle
from ocr_mllm_toy.model.builder import load_pretrained_model
from ocr_mllm_toy.conversation import conv_templates, SeparatorStyle
from ocr_mllm_toy.model import *
from ocr_mllm_toy.utils import *
from ocr_mllm_toy.mm_utils import *

LOGDIR = "./ocr_mmlm_toy_gradio/gradio_log"

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'

headers = {"User-Agent": "OCR_MLLM_TOY Client"}
no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)

priority = ["./checkpoints/"]

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")

class ModelWorker:
    def __init__(self,model_path,load_8bit, load_4bit, device):
        disable_torch_init()
        model_name = os.path.expanduser(model_path)
        self.tokenizer, self.model, self.image_processor, self.image_processor_high, context_len = load_pretrained_model(model_name, model_name, load_8bit, load_4bit, device=device)
        self.device = device
        self.is_multimodal = True

    @torch.inference_mode()
    def generate_stream(self, params):   ###模型生成文本的主要函数，处理prompt
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor
        prompt = params["prompt"]
        ori_prompt = prompt
        images = params.get("images", None)
        num_image_tokens = 0
        if images is not None and len(images) > 0 and self.is_multimodal:
            if len(images) > 0:
                if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
                    raise ValueError("Number of images does not match number of <image> tokens in prompt")

                images = [load_image_from_base64(image) for image in images]
                images_tensor = self.image_processor.preprocess(images[0], return_tensors='pt')['pixel_values'][0]
                images_tensor_high = self.image_processor_high(images[0])

                if type(images_tensor) is list:
                    images_tensor = [image_tensor.to(self.model.device, dtype=torch.float16) for image_tensor in images_tensor]
                    images_tensor_high = [image_tensor.to(self.model.device, dtype=torch.float16) for image_tensor in images_tensor_high]
                else:
                    images_tensor = images_tensor.unsqueeze(0).half().to(self.model.device)
                    images_tensor_high = images_tensor_high.unsqueeze(0).half().to(self.model.device)

                replace_token = DEFAULT_IMAGE_TOKEN
                # if self.use_im_start_end:
                #     # replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                #     replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN*256 + DEFAULT_IM_END_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
                print(f"当前的prompt是:{prompt}\n")
                num_image_tokens = prompt.count(replace_token) * 256
                # num_image_tokens = prompt.count(replace_token) * model.get_vision_tower().num_patches
            else:
                images = None
        else:
            images = None
            images_tensor = None
            images_tensor_high = None
        ###设置超参数
        temperature = float(params.get("temperature", 1.0))
        repetition_penalty = float(params.get("repetition_penalty", 1.1))
        top_p = float(params.get("top_p", 1.0))
        max_context_length = getattr(model.config, 'max_position_embeddings', 2048)
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        stop_str = params.get("stop", None)
        do_sample = True if temperature > 0.001 else False

        images_tensor = images_tensor
        images_tensor_high = images_tensor_high
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        # inputs = self.tokenizer([prompt])
        # input_ids = torch.as_tensor(inputs.input_ids).to(self.device)
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

        max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)

        if max_new_tokens < 1:
            yield json.dumps({"text": ori_prompt + "Exceeds max token length. Please start a new conversation, thanks.", "error_code": 0}).encode() + b"\0"
            return

        thread = Thread(target=model.generate, kwargs=dict(
            inputs=input_ids,
            images=images_tensor,
            images_high=images_tensor_high,
            do_sample=do_sample,
            temperature=temperature,
            top_k=0,
            top_p=top_p,
            repetition_penalty=repetition_penalty,   ###input_id中的image token为-200，但是再repetition_penalty中（RepetitionPenaltyLogitsProcessor）input_id不能为负数
            num_beams=1,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            stopping_criteria=[stopping_criteria],
            use_cache=True,
        ))
        thread.start()

        generated_text = ori_prompt
        for new_text in streamer:
            generated_text += new_text
            if generated_text.endswith(stop_str):
                generated_text = generated_text[:-len(stop_str)]
            yield json.dumps({"text": generated_text, "error_code": 0}).encode() + b"\0"

    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
        except ValueError as e:
            print("Caught ValueError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.CudaError as e:
            print("Caught torch.cuda.CudaError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            print("Caught Unknown Error", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name

def get_model_list():
    return priority

def load_demo_refresh_model_list():
    models = get_model_list()
    state = conv_templates[args.conv_mode].copy()
    dropdown_update = gr.Dropdown.update(
        choices=models,
        value=models[0] if len(models) > 0 else ""
    )
    return state, dropdown_update

def regenerate(state, image_process_mode, request: gr.Request):
    torch.cuda.empty_cache()
    logger.info(f"regenerate. ip: {request.client.host}")
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5

def clear_history(request: gr.Request):
    torch.cuda.empty_cache()
    logger.info(f"clear_history. ip: {request.client.host}")
    state = default_conversation.copy()
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5

def add_text(state, text, image, image_process_mode, request: gr.Request):
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    if len(text) <= 0 and image is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 5
    if args.moderate:  ###输入文字审核
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            return (state, state.to_gradio_chatbot(), moderation_msg, None) + (
                no_change_btn,) * 5

    text = text[:2048]  # Hard cut-off
    if image is not None:
        text = text[:1748]  # Hard cut-off for images
        if '<image>' not in text:
            text = '<image>\n' + text
            # text = text + '\n<image>'
        text = (text, image, image_process_mode)
        if len(state.get_images(return_pil=True)) > 0:
            state = default_conversation.copy()
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def http_bot(state, model_selector, temperature,repetition_penalty, top_p, max_new_tokens, request: gr.Request):
    start_tstamp = time.time()
    model_name = model_selector

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    if len(state.messages) == state.offset + 2:
        new_state = conv_templates[args.conv_mode].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Construct prompt
    prompt = state.get_prompt()
    all_images = state.get_images(return_pil=True)
    all_image_hash = [hashlib.md5(image.tobytes()).hexdigest() for image in all_images]
    for image, hash in zip(all_images, all_image_hash):
        t = datetime.datetime.now()
        filename = os.path.join(LOGDIR, "serve_images", f"{t.year}-{t.month:02d}-{t.day:02d}", f"{hash}.jpg")
        if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            image.save(filename)

    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": float(temperature),
        "repetition_penalty": float(repetition_penalty),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2,
        "images": f'List of {len(state.get_images())} images: {all_image_hash}',
    }
    logger.info(f"==== request ====\n{pload}")
    pload['images'] = state.get_images()
    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    try:
        for chunk in worker.generate_stream_gate(pload):
            if chunk:
                data = json.loads(str(chunk, encoding = "utf-8").strip("\0"))   ###获得模型生成的结果
                # data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt):].strip()
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
                    return
                time.sleep(0.03)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state.dict(),
            "images": all_image_hash,
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")



title_markdown = ("""
#  OCR_MLLM_TOY: Large Language and Vision Model
""")
block_css = """
#buttons button {
    min-width: min(120px,100%);
}
"""

def build_demo(embed_mode):
    textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)
    with gr.Blocks(title="OCR_MLLM_TOY", theme=gr.themes.Default(), css=block_css) as demo:
        state = gr.State()
        if not embed_mode:
            gr.Markdown(title_markdown)
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row(elem_id="model_selector_row"):
                    model_selector = gr.Dropdown(
                        choices=models,
                        value=models[0] if len(models) > 0 else "",
                        interactive=True,
                        show_label=False,
                        container=False)
                imagebox = gr.Image(type="pil")
                image_process_mode = gr.Radio(
                    ["Crop", "Resize", "Pad", "Default"],
                    value="Default",
                    label="Preprocess for non-square image", visible=False)  ###对上传图片的处理方式

                cur_dir = os.path.dirname(os.path.abspath(__file__))
                gr.Examples(examples=[
                    [f"{cur_dir}/examples/qwen.png", "who are the authors of this paper?"],
                    [f"{cur_dir}/examples/fanti2.png", "图中有哪些文字。"],
                    [f"{cur_dir}/examples/mulu.png", "图中有哪些文字。"],
                    [f"{cur_dir}/examples/test1.png", "新能源重卡充电设备主要分为哪些类？"],
                ], inputs=[imagebox, textbox])

                with gr.Accordion("Parameters", open=False) as parameter_row:
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.1, interactive=True, label="Temperature",)
                    repetition_penalty = gr.Slider(minimum=1.0, maximum=2.0, value=1.1, step=0.1, interactive=True, label="repetition_penalty",)
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.8, step=0.1, interactive=True, label="Top P",)
                    max_output_tokens = gr.Slider(minimum=0, maximum=2048, value=2048, step=64, interactive=True, label="Max output tokens",)

            with gr.Column(scale=8):
                chatbot = gr.Chatbot(elem_id="chatbot", label="OCR_MLLM_TOY Chatbot", height=550)
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(value="Send", variant="primary")
                with gr.Row(elem_id="buttons") as button_row:
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️  Clear", interactive=False)

        btn_list = [regenerate_btn, clear_btn]  ###button list

        regenerate_btn.click(
            regenerate,
            [state, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        ).then(
            http_bot,
            [state, model_selector, temperature, repetition_penalty, top_p, max_output_tokens],
            [state, chatbot] + btn_list
        )

        clear_btn.click(
            clear_history,
            None,
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        )

        textbox.submit(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        ).then(
            http_bot,
            [state, model_selector, temperature, repetition_penalty, top_p, max_output_tokens],
            [state, chatbot] + btn_list
        )

        submit_btn.click(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        ).then(
            http_bot,
            [state, model_selector, temperature, repetition_penalty, top_p, max_output_tokens],
            [state, chatbot] + btn_list
        )
        demo.load(
            load_demo_refresh_model_list,
            None,
            [state, model_selector],
            queue=False
        )
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="122.226.240.94")
    parser.add_argument("--port", type=int, default=7866)
    parser.add_argument("--concurrency-count", type=int, default=10)

    parser.add_argument("--share", type=bool,default=True)
    parser.add_argument("--moderate", type=bool,default=False)
    parser.add_argument("--embed", type=bool,default=False)

    parser.add_argument("--model-path", type=str, default="./checkpoints/qwen14b-finetune_all/checkpoint-8300")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="mpt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--load-8bit", type=bool, default=True)
    parser.add_argument("--load-4bit", type=bool, default=False)
    args = parser.parse_args()
    logger.info(f"args: {args}")
    default_conversation = conv_templates[args.conv_mode]
    worker = ModelWorker(args.model_path, args.load_8bit, args.load_4bit, args.device)

    models = get_model_list()
    demo = build_demo(args.embed)
    demo.queue(
        concurrency_count=args.concurrency_count,
        api_open=False
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )
