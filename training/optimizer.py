import torch
from torch import nn


def set_training_stage(model, epoch):
    vit = model.frequency.vit
    vit.requires_grad_(False)
    if epoch >= 3:
        for block in list(vit.blocks)[-4:]:
            block.requires_grad_(True)
        vit.norm.requires_grad_(True)
    if epoch >= 6:
        vit.requires_grad_(True)
    if hasattr(model, "spatial"):
        backbone = model.spatial.backbone
        backbone.requires_grad_(False)
        if epoch >= 3:
            for stage in list(backbone.blocks)[-2:]:
                stage.requires_grad_(True)
            # Proyección final del backbone, además de las últimas dos etapas MBConv.
            backbone.conv_head.requires_grad_(True)
        if model.spatial.freeze_batchnorm:
            for layer in backbone.modules():
                if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                    layer.eval()
                    layer.requires_grad_(False)
    model.frequency.lifting.requires_grad_(True)


def build_optimizer(model, config):
    t = config["training"]
    grouped = {}
    count = 0
    for name, parameter in model.named_parameters():
        if name.startswith("frequency.lifting."):
            category = "lifting"
        elif name.startswith("frequency.patch_embedding."):
            category = "patch_embedding"
        elif name.startswith("frequency.vit."):
            category = "vit"
        elif name.startswith("spatial.backbone."):
            category = "efficientnet"
        elif name.startswith("spatial."):
            category = "spatial_projection"
        elif name.startswith("cross_attention."):
            category = "cross_attention"
        elif name.startswith("classifier."):
            category = "classifier"
        else:
            raise ValueError(f"Parámetro sin grupo: {name}")
        no_decay = category == "lifting" or parameter.ndim < 2 or any(token in name for token in ("cls_token", "pos_embed", "position"))
        key = category, no_decay
        if key not in grouped:
            grouped[key] = {"params": [], "lr": t[f"lr_{category}"], "weight_decay": 0.0 if no_decay else t["weight_decay"], "name": f"{category}/{'no_decay' if no_decay else 'decay'}"}
        grouped[key]["params"].append(parameter)
        count += parameter.numel()
    if count != sum(p.numel() for p in model.parameters()):
        raise RuntimeError("Optimizador incompleto")
    # Incluye también los pesos temporalmente congelados; no se reconstruye al descongelar.
    return torch.optim.AdamW(list(grouped.values()))
