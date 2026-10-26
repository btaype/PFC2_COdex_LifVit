import hashlib
import logging
from pathlib import Path

import torch


def cached_public_checkpoint(model_name, cache_dir):
    import timm
    from huggingface_hub import try_to_load_from_cache
    cfg = timm.models.get_pretrained_cfg(model_name)
    if not cfg or not cfg.hf_hub_id or not cache_dir:
        return None
    repo, _, revision = cfg.hf_hub_id.partition("@")
    for filename in ("model.safetensors", "pytorch_model.bin"):
        path = try_to_load_from_cache(repo, filename, cache_dir=str(cache_dir), revision=revision or "main")
        if isinstance(path, str) and Path(path).is_file():
            return Path(path)
    return None


def public_model_kwargs(model_name, cache_dir):
    """Usa el archivo de la fuente pública elegida cuando ya está en caché, sin red."""
    result = {"cache_dir": str(cache_dir)}
    cached = cached_public_checkpoint(model_name, cache_dir)
    if cached:
        # ViT AugReg publica también NPZ; el archivo del Hub ya es state_dict convertido.
        result["pretrained_cfg_overlay"] = {"file": str(cached), "custom_load": False}
    return result


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_backbone(model, path, excluded):
    """Checkpoint timm state_dict. Se rechazan cargas parciales silenciosas."""
    obj = torch.load(path, map_location="cpu", weights_only=True)
    state = obj.get("state_dict", obj.get("model_state_dict", obj.get("model", obj)))
    state = {k.removeprefix("module."): v for k, v in state.items()}
    expected = {k: v for k, v in model.state_dict().items() if not k.startswith(excluded)}
    missing = [k for k in expected if k not in state]
    mismatched = [k for k in expected if k in state and state[k].shape != expected[k].shape]
    unexpected = [k for k in state if k not in expected and not k.startswith(excluded)]
    if missing or mismatched or unexpected:
        raise ValueError(f"Checkpoint incompatible. Faltan={missing}; dimensiones={mismatched}; desconocidos={unexpected}")
    omitted = sorted(k for k in state if k.startswith(excluded))
    model.load_state_dict({k: state[k] for k in expected}, strict=False)
    logging.info("Pesos cargados: %s; omitidos: %s", sorted(expected), omitted)
    return {"path": str(Path(path).resolve()), "sha256": file_hash(path), "loaded": sorted(expected), "omitted": omitted}


def source_info(model, options, initialize, excluded, cache_dir=None):
    if initialize and options.get("checkpoint_path"):
        return load_backbone(model, options["checkpoint_path"], excluded)
    if initialize and options.get("pretrained"):
        cfg = model.pretrained_cfg
        source = {"library": "timm", "model_name": options["model_name"], "url": cfg.get("url"), "hf_hub_id": cfg.get("hf_hub_id"), "tag": cfg.get("tag")}
        cached = cached_public_checkpoint(options["model_name"], cache_dir)
        if cached:
            source.update(weights_file=str(cached), weights_sha256=file_hash(cached), hub_revision=cached.parent.name)
        return source
    return {"initialization": "random" if initialize else "training_checkpoint"}


def check_public_normalization(model, mean, std, public):
    if public:
        cfg = model.pretrained_cfg
        if not torch.allclose(torch.tensor(mean), torch.tensor(cfg["mean"]), atol=1e-6) or not torch.allclose(torch.tensor(std), torch.tensor(cfg["std"]), atol=1e-6):
            raise ValueError(f"Normalización incompatible con el checkpoint público: usa mean={cfg['mean']} std={cfg['std']}")
