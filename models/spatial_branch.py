import logging

import timm
import torch
from torch import nn

from utils.config import resolve

from .weights import check_public_normalization, public_model_kwargs, source_info


class SpatialBranch(nn.Module):
    def __init__(self, config, initialize=True):
        super().__init__()
        options = config["efficientnet"]
        if options["model_name"].split(".")[0] not in ("efficientnet_b4", "tf_efficientnet_b4"):
            raise ValueError("Se requiere EfficientNet-B4")
        public = initialize and options["pretrained"] and not options.get("checkpoint_path")
        if public:
            logging.info("Cargando pesos públicos elegidos: %s", options["model_name"])
        cache = resolve(config.get("pretrained_cache_dir", "checkpoints/pretrained/timm"))
        self.backbone = timm.create_model(options["model_name"], pretrained=public, num_classes=0,
                                          **(public_model_kwargs(options["model_name"], cache) if public else {}))
        self.source = source_info(self.backbone, options, initialize, ("classifier.",), cache_dir=cache)
        self.projection = nn.Conv2d(1792, 768, kernel_size=1)
        self.position = nn.Parameter(torch.zeros(1, 49, 768))
        nn.init.trunc_normal_(self.position, std=0.02)
        prep = config["preprocessing"]
        check_public_normalization(self.backbone, prep["efficientnet_mean"], prep["efficientnet_std"], public)
        self.register_buffer("mean", torch.tensor(prep["efficientnet_mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(prep["efficientnet_std"]).view(1, 3, 1, 1))
        self.freeze_batchnorm = options["freeze_batchnorm"]

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_batchnorm:
            for layer in self.backbone.modules():
                if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                    layer.eval()
                    layer.requires_grad_(False)
        return self

    def forward(self, x):
        features = self.backbone.forward_features((x - self.mean) / self.std)
        if features.shape[1:] != (1792, 7, 7):
            raise RuntimeError(f"Mapa EfficientNet inesperado: {features.shape}")
        return self.projection(features).flatten(2).transpose(1, 2) + self.position
