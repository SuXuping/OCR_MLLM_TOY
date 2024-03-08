from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

import math
from functools import partial

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector
from .multimodal_encoder.sam import build_sam_vit_b
from ocr_mllm_toy.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from .multimodal_encoder.qwen_clip import Resampler
from .multimodal_encoder.utils import train_transform,test_transform
from .multimodal_encoder.clip_encoder import CLIPVisionTower,Qwen_CLIPVisionTower

###qwen
from ocr_mllm_toy.model.language_model.qwen14b.modeling_qwen import QWenLMHeadModel,QWenModel
from ocr_mllm_toy.model.language_model.qwen14b.configuration_qwen import QWenConfig

class OCR_MLLM_TOYQwenConfig(QWenConfig):
    model_type = "OCR_MLLM_TOY_Qwen"

class OCR_MLLM_TOYQWenModel(QWenModel):  ###BaichuanModel只含有decoder的架构，OCR_MLLM_TOYMetaModel包含vision架构，所以此类是用于初始化整个vision和llm的decoder layer的模型
    config_class = OCR_MLLM_TOYQwenConfig

    def __init__(self, config: QWenConfig):  
        super(OCR_MLLM_TOYQWenModel, self).__init__(config)
        # if hasattr(config,"mm_vision_tower"):
        #     print('初始化视觉模型')
        #     # self.vision_tower = Qwen_CLIPVisionTower("./ocr_mllm_toy/pretrain_weight/qwen_vit_448")
        #     self.vision_tower = build_vision_tower(config, delay_load=True)

        #     norm_layer = partial(nn.LayerNorm, eps=1e-6)
        #     self.mm_projector = nn.Sequential(
        #         Resampler(
        #         grid_size=int(math.sqrt(256)),
        #         embed_dim=5120,
        #         num_heads=5120 // 128,
        #         kv_dim=1664,
        #         norm_layer=norm_layer,),
        #     norm_layer(5120)
        #     )   ###cross atten的方式进行连接

        #     # self.vision_tower_high = build_sam_vit_b("./ocr_mllm_toy/pretrain_weight/vary_pretrain/vision_tower_high.bin")
        #     self.vision_tower_high = build_sam_vit_b()
        #     self.vision_tower_high.image_processor_high = train_transform
        #     self.config.mm_hidden_size = self.vision_tower_high.hidden_size  ##vision_tower_high最后输出的维度
        #     self.mm_projector_high = build_vision_projector(config)

    # def get_embed_tokens(self):
    #     return self.get_input_embeddings()

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    def get_vision_tower_high(self):
        vision_tower_high = getattr(self, 'vision_tower_high', None)  
        if type(vision_tower_high) is list:
            vision_tower_high = vision_tower_high[0]
        return vision_tower_high
    
    def initialize_vision_modules(self, model_args, fsdp=None):  ###将配置文件赋值为config文件，用于后面的保存
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            # vision_tower = Qwen_CLIPVisionTower("./ocr_mllm_toy/pretrain_weight/qwen_vit_448")
            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        if self.get_vision_tower_high() is None:
            vision_tower_high = build_sam_vit_b("./ocr_mllm_toy/pretrain_weight/vary_pretrain/vision_tower_high.bin")
            vision_tower_high.image_processor_high = train_transform
            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower_high = [vision_tower_high]
            else:
                self.vision_tower_high = vision_tower_high
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower_high = self.vision_tower_high[0]
            else:
                vision_tower_high = self.vision_tower_high
            state_dict = torch.load("./ocr_mllm_toy/pretrain_weight/vary_pretrain/vision_tower_high.bin")
            vision_tower_high.load_state_dict(state_dict, strict=True)

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower_high.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
            self.mm_projector = nn.Sequential(
                Resampler(
                grid_size=int(math.sqrt(256)),
                embed_dim=5120,
                num_heads=5120 // 128,
                kv_dim=1664,
                norm_layer=norm_layer,),
            norm_layer(5120)
            )   ###cross atten的方式进行连接
        else:
            # In case it is frozen by LoRA
            print('训练映射层')
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if getattr(self, 'mm_projector_high', None) is None:
            self.mm_projector_high = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            print('训练映射层')
            for p in self.mm_projector_high.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            self.mm_projector_high.load_state_dict(get_w(mm_projector_weights, 'mm_projector_high'))

    def initialize_vision_weights(self, model_args, fsdp=None):  ###将配置文件赋值为config文件，用于后面的保存
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.vision_tower = build_vision_tower(model_args)
        self.vision_tower_high = build_sam_vit_b("./ocr_mllm_toy/pretrain_weight/vary_pretrain/vision_tower_high.bin")
        self.vision_tower_high.image_processor_high = train_transform

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.mm_projector = nn.Sequential(
            Resampler(
            grid_size=int(math.sqrt(256)),
            embed_dim=5120,
            num_heads=5120 // 128,
            kv_dim=1664,
            norm_layer=norm_layer,),
        norm_layer(5120)
        )   ###cross atten的方式进行连接

        self.mm_projector_high = build_vision_projector(self.config)

        mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
        print(f'映射层：\n {mm_projector_weights.keys()}')

        projector_weigth = {}
        projector_weigth_high = {}
        for k,v in mm_projector_weights.items():
            if 'mm_projector_high' in k:
                projector_weigth_high[k.split('mm_projector_high' + '.')[1]] = v
            if 'mm_projector' in k and not 'mm_projector_high' in k:
                projector_weigth[k.split('mm_projector' + '.')[1]] = v
        self.mm_projector.load_state_dict(projector_weigth)
        self.mm_projector_high.load_state_dict(projector_weigth_high)

class OCR_MLLM_TOYQWenForCausalLM(QWenLMHeadModel):  
    config_class = OCR_MLLM_TOYQwenConfig

    def __init__(self, config):
        super(OCR_MLLM_TOYQWenForCausalLM, self).__init__(config)
        self.transformer = OCR_MLLM_TOYQWenModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.transformer

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_high: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal_qwen(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                images_high
            )

        return super().forward(   ###继承QWenLMHeadModel的forward
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_vision_tower_high(self):
        return self.get_model().get_vision_tower_high()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        for n,p in self.get_model().mm_projector.named_parameters():
            dtype = p.dtype
            device = p.device
        image_features = self.get_model().mm_projector(image_features.to(device=device, dtype=dtype))
        return image_features

    def encode_images_high(self, images_high):
        for n,p in self.get_model().get_vision_tower_high().named_parameters():
            dtype = p.dtype
            device = p.device
        image_features = self.get_model().get_vision_tower_high()(images_high.to(device=device, dtype=dtype))
        image_features = image_features.flatten(2).permute(0, 2, 1)
        for n,p in self.get_model().mm_projector_high.named_parameters():
            dtype = p.dtype
            device = p.device
        image_features = self.get_model().mm_projector_high(image_features.to(device=device, dtype=dtype))
        return image_features

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:   ###如果使用mm_use_im_start_end，需要将llm的embedding层放开
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

    def prepare_inputs_labels_for_multimodal_qwen(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images,images_high
    ):
        vision_tower = self.get_vision_tower()
        vision_tower_high = self.get_vision_tower_high()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-3] + 1   ###qwen的key和value是(B,L,40,128)所以是-3,百川的是(B,40,L,128)所以是-2
                # if target_shape - attention_mask.shape[1] > 0:
                #     attention_mask = torch.cat((attention_mask, torch.ones(
                #         (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                #         dtype=attention_mask.dtype,
                #         device=attention_mask.device
                #     )), dim=1)
                # else:
                #     print(f'target_shape == {target_shape},  attention_mask == {attention_mask.shape}')
                attention_mask = torch.ones(
                    (attention_mask.shape[0], target_shape),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features_low = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features_low = torch.split(image_features_low, split_sizes, dim=0)
            image_features_low = [x.flatten(0, 1).to(self.device) for x in image_features_low]
        else:
            image_features_low = self.encode_images(images).to(self.device)  ###([8, 3, 448, 448])-->([8, 256, 5120])

        if type(images_high) is list or images_high.ndim == 5:
            concat_images = torch.cat([image for image in images_high], dim=0)
            image_features_high = self.encode_images_high(concat_images)
            split_sizes = [image.shape[0] for image in images_high]
            image_features_high = torch.split(image_features_high, split_sizes, dim=0)
            image_features_high = [x.flatten(0, 1).to(self.device) for x in image_features_high]
        else:
            image_features_high = self.encode_images_high(images_high).to(self.device)  ###([8, 3, 448, 448])-->([8, 256, 5120])

        ###将image_features  cat起来
        image_features = torch.cat((image_features_high, image_features_low), dim=1)
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):  ###将图片embedding([576, 5120])加入
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        images_high = kwargs.pop("images_high", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
            _inputs['images_high'] = images_high
        return _inputs

AutoConfig.register("OCR_MLLM_TOY_Qwen", OCR_MLLM_TOYQwenConfig)
AutoModelForCausalLM.register(OCR_MLLM_TOYQwenConfig, OCR_MLLM_TOYQWenForCausalLM)