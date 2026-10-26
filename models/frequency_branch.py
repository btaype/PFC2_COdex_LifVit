import logging

import timm
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from utils.config import resolve

from .learnable_lifting import LearnableLifting2D
from .weights import check_public_normalization, public_model_kwargs, source_info


class FrequencyBranch(nn.Module):
    def __init__(self, config, initialize=True):
        super().__init__()
        options = config["vit"]
        if options["model_name"].split(".")[0] != "vit_base_patch16_224":
            raise ValueError("Se requiere ViT-B/16 de 224, sin tokens adicionales")
        public = initialize and options["pretrained"] and not options.get("checkpoint_path")
        if public:
            logging.info("Cargando pesos públicos elegidos: %s", options["model_name"])
        cache = resolve(config.get("pretrained_cache_dir", "checkpoints/pretrained/timm"))
        self.vit = timm.create_model(options["model_name"], pretrained=public, num_classes=0,
                                     **(public_model_kwargs(options["model_name"], cache) if public else {}))
        if self.vit.embed_dim != 768 or len(self.vit.blocks) != 12 or self.vit.pos_embed.shape != (1, 197, 768):
            raise ValueError("Backbone ViT incompatible con la tesis")
        self.source = source_info(self.vit, options, initialize, ("head.", "patch_embed."), cache_dir=cache)
        self.vit.patch_embed = nn.Identity()
        self.lifting = LearnableLifting2D()
        self.patch_embedding = nn.Conv2d(12, 768, kernel_size=8, stride=8)
        prep = config["preprocessing"]
        check_public_normalization(self.vit, prep["vit_mean"], prep["vit_std"], public)
        self.register_buffer("mean", torch.tensor(prep["vit_mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(prep["vit_std"]).view(1, 3, 1, 1))
        self.gradient_checkpointing = config["training"]["gradient_checkpointing"]

    def forward(self, x):
        bands = self.lifting((x - self.mean) / self.std)
        patches = self.patch_embedding(bands).flatten(2).transpose(1, 2)
        tokens = torch.cat((self.vit.cls_token.expand(x.shape[0], -1, -1), patches), dim=1)
        tokens = self.vit.norm_pre(self.vit.pos_drop(tokens + self.vit.pos_embed))
        for block in self.vit.blocks:
            tokens = checkpoint(block, tokens, use_reentrant=False) if self.training and self.gradient_checkpointing else block(tokens)
        return self.vit.norm(tokens)
