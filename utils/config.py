from copy import deepcopy
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ("liftvit", "liftvit_h")


def merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def read_yaml(path, seen=None):
    path = Path(path).resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Herencia circular de configuración: {path}")
    seen.add(path)
    with path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    parent = cfg.pop("extends", None)
    return merge(read_yaml(path.parent / parent, seen), cfg) if parent else cfg


def load_config(path=None, model=None):
    cfg = read_yaml(path or ROOT / "configs" / f"{model or 'liftvit'}.yaml")
    selected = model or cfg["model"]
    if selected not in MODEL_NAMES:
        raise ValueError(f"Solo se admiten {MODEL_NAMES}")
    # Los perfiles específicos se aplican antes del protocolo cuando se cambia de modelo.
    if selected != cfg["model"]:
        cfg = merge(read_yaml(ROOT / "configs" / f"{selected}.yaml"), cfg)
        profile = read_yaml(ROOT / "configs" / f"{selected}.yaml")["training"]
        for key in ("batch_size", "gradient_accumulation_steps", "lr_lifting", "lr_patch_embedding", "lr_classifier"):
            cfg["training"][key] = profile[key]
    cfg["model"] = selected
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    name, t = cfg["model"], cfg["training"]
    if name not in MODEL_NAMES:
        raise ValueError("Arquitectura no permitida")
    expected = 10 if name == "liftvit" else 4
    if t["batch_size"] != expected:
        raise ValueError(f"{name} requiere batch físico {expected}; no se ajusta automáticamente")
    if cfg["preprocessing"]["image_size"] != 224:
        raise ValueError("La entrada arquitectónica es 224x224")
    if any(not isinstance(t[k], int) or t[k] < 1 for k in ("gradient_accumulation_steps", "epochs", "checkpoint_interval")):
        raise ValueError("Acumulación y épocas deben ser positivas")
    if t.get("device", "auto") not in ("auto", "cpu", "cuda"):
        raise ValueError("Dispositivo inválido")
    if not 0 <= t["warmup_fraction"] <= 1 or not 0 <= t["min_lr_factor"] <= 1:
        raise ValueError("Warmup y factor LR mínimo deben estar en [0,1]")
    if cfg["protocol"] not in ("diffusion_training", "ffpp_training"):
        raise ValueError("Protocolo desconocido")
    if cfg["protocol"] == "diffusion_training":
        if set(cfg["data"]["train_generators"]) != {"ddpm", "ldm"} or set(cfg["data"]["validation_generators"]) != {"ddpm", "ldm"}:
            raise ValueError("El protocolo principal usa únicamente DDPM/LDM en train y val")
    for prefix in ("vit", "efficientnet"):
        mean, std = cfg["preprocessing"][f"{prefix}_mean"], cfg["preprocessing"][f"{prefix}_std"]
        if len(mean) != 3 or len(std) != 3 or any(not math.isfinite(s) for s in mean + std) or any(s <= 0 for s in std):
            raise ValueError("Estadísticas RGB inválidas")


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path
