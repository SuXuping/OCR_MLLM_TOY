import torch
import torch.nn as nn
import os

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from ocr_mllm_toy.model.multimodal_encoder.qwen_clip import VisionTransformer

# class CLIPVisionTower(nn.Module):
#     def __init__(self, vision_tower, args, delay_load=False):
#         super().__init__()

#         self.is_loaded = False

#         self.vision_tower_name = vision_tower
#         self.select_layer = args.mm_vision_select_layer
#         self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

#         if not delay_load:
#             self.load_model()
#         else:
#             self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

#     def load_model(self):
#         self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
#         self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
#         self.vision_tower.requires_grad_(False)

#         self.is_loaded = True

#     def feature_select(self, image_forward_outs):
#         image_features = image_forward_outs.hidden_states[self.select_layer]
#         if self.select_feature == 'patch':
#             image_features = image_features[:, 1:]
#         elif self.select_feature == 'cls_patch':
#             image_features = image_features
#         else:
#             raise ValueError(f'Unexpected select feature: {self.select_feature}')
#         return image_features

#     @torch.no_grad()
#     def forward(self, images):
#         if type(images) is list:
#             image_features = []
#             for image in images:
#                 image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
#                 image_feature = self.feature_select(image_forward_out).to(image.dtype)
#                 image_features.append(image_feature)
#         else:
#             image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
#             image_features = self.feature_select(image_forward_outs).to(images.dtype)

#         return image_features

#     @property
#     def dummy_feature(self):
#         return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

#     @property
#     def dtype(self):
#         return self.vision_tower.dtype

#     @property
#     def device(self):
#         return self.vision_tower.device

#     @property
#     def config(self):
#         if self.is_loaded:
#             return self.vision_tower.config
#         else:
#             return self.cfg_only

#     @property
#     def hidden_size(self):
#         return self.config.hidden_size

#     @property
#     def num_patches(self):
#         return (self.config.image_size // self.config.patch_size) ** 2


# class Qwen_CLIPVisionTower(nn.Module):
#     def __init__(self, vision_tower):
#         super().__init__()
#         self.is_loaded = True
#         self.vision_tower_name = vision_tower
#         self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)

#         qwen_config = AutoConfig.from_pretrained(self.vision_tower_name)
#         self.vis_config = qwen_config.vision_config
#         self.vision_tower = VisionTransformer(
#             image_size=448,
#             patch_size=self.vis_config.patch_size,
#             width=self.vis_config.hidden_size,
#             layers=self.vis_config.num_hidden_layers,
#             heads=self.vis_config.num_attention_heads,
#             mlp_size=self.vis_config.intermediate_size,
#             output_dim=5120,
#         ) 
#         self.vision_tower.load_state_dict(torch.load(os.path.join(self.vision_tower_name, 'pytorch_model.bin'), map_location='cpu'), strict=False)
#         # self.vis_config.hidden_size = 5120
#         for name,param in self.vision_tower.named_parameters():
#             self.vision_tower.device = param.device
#             self.vision_tower.dtype = param.dtype

#         self.vision_tower.requires_grad_(False)

#     def feature_select(self, image_forward_outs):
#         # image_features = image_forward_outs.hidden_states[self.select_layer]
#         # if self.select_feature == 'patch':
#         #     image_features = image_features[:, 1:]
#         # elif self.select_feature == 'cls_patch':
#         #     image_features = image_features
#         # else:
#         #     raise ValueError(f'Unexpected select feature: {self.select_feature}')
#         image_features = image_forward_outs
#         return image_features

#     @torch.no_grad()
#     def forward(self, images):
#         if type(images) is list:
#             image_features = []
#             for image in images:
#                 image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
#                 image_feature = self.feature_select(image_forward_out).to(image.dtype)
#                 image_features.append(image_feature)
#         else:
#             image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
#             image_features = self.feature_select(image_forward_outs).to(images.dtype)

#         return image_features

#     @property
#     def dummy_feature(self):
#         return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

#     @property
#     def dtype(self):
#         for name,param in self.vision_tower.named_parameters():
#             self.vision_tower.dtype = param.dtype
#         return self.vision_tower.dtype

#     @property
#     def device(self):
#         for name,param in self.vision_tower.named_parameters():
#             self.vision_tower.device = param.device
#         return self.vision_tower.device

#     @property
#     def config(self):
#         if self.is_loaded:
#             # return self.vision_tower.config
#             return self.vis_config
#         else:
#             return self.cfg_only

#     @property
#     def hidden_size(self):
#         return self.config.hidden_size

#     @property
#     def num_patches(self):
#         return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class Qwen_CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)

        qwen_config = AutoConfig.from_pretrained(self.vision_tower_name)
        self.vis_config = qwen_config.vision_config
        self.vision_tower = VisionTransformer(
            image_size=448,
            patch_size=self.vis_config.patch_size,
            width=self.vis_config.hidden_size,
            layers=self.vis_config.num_hidden_layers,
            heads=self.vis_config.num_attention_heads,
            mlp_size=self.vis_config.intermediate_size,
            output_dim=5120,
        ) 
        self.vision_tower.load_state_dict(torch.load(os.path.join(self.vision_tower_name, 'pytorch_model.bin'), map_location='cpu'), strict=False)
        self.vis_config.hidden_size = 5120      
        for name,param in self.vision_tower.named_parameters():
            self.vision_tower.device = param.device
            self.vision_tower.dtype = param.dtype
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        # image_features = image_forward_outs.hidden_states[self.select_layer]
        # if self.select_feature == 'patch':
        #     image_features = image_features[:, 1:]
        # elif self.select_feature == 'cls_patch':
        #     image_features = image_features
        # else:
        #     raise ValueError(f'Unexpected select feature: {self.select_feature}')
        image_features = image_forward_outs
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            # return self.vision_tower.config
            return self.vis_config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
