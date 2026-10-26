"""Comprueba los pesos públicos elegidos y Haar inicial; no requiere ni genera datasets."""
import argparse
import gc
import logging

import torch

from models import build_model
from models.weights import file_hash
from utils.config import MODEL_NAMES, load_config, resolve
from utils.reproducibility import save_json, seed_everything


def cached_files(cfg, model_name):
    cache = resolve(cfg["pretrained_cache_dir"])
    repo = cache / ("models--timm--" + model_name)
    return [{"path": str(path.relative_to(cache)), "sha256": file_hash(path)}
            for path in sorted(repo.glob("snapshots/*/*")) if path.suffix in (".safetensors", ".bin", ".pth")]


def verify(name):
    cfg = load_config(model=name)
    seed_everything(cfg["seed"])
    model = build_model(cfg, initialize=True).eval()
    kernels = model.frequency.lifting.effective_kernels()
    for key, value in kernels.items():
        expected = torch.zeros(3, 1, 3, 3)
        expected[:, :, 1, 1] = 1.0 if key.startswith("P") else 0.5
        torch.testing.assert_close(value, expected, atol=0, rtol=0)
    # Solo un tensor nulo para comprobar dimensiones, sin crear archivos de imágenes.
    tensor = torch.zeros(1, 3, 224, 224)
    with torch.inference_mode():
        frequency = model.frequency(tensor)
        assert frequency.shape == (1, 197, 768)
        if hasattr(model, "spatial"):
            features = model.spatial.backbone.forward_features((tensor - model.spatial.mean) / model.spatial.std)
            assert features.shape == (1, 1792, 7, 7)
        logits = model(tensor)
        assert logits.shape == (1, 1) and torch.isfinite(logits).all()
    sources = {"vit": model.frequency.source}
    if hasattr(model, "spatial"):
        sources["efficientnet"] = model.spatial.source
    for key in sources:
        sources[key]["cache_files"] = cached_files(cfg, cfg[key]["model_name"])
    result = {"model": name, "public_weights_loaded": True, "sources": sources,
              "datasets_used": False, "test_input": "zero_tensor_for_dimensions_only", "logits_shape": list(logits.shape),
              "frequency_tokens": list(frequency.shape), "haar_initialization_preserved": True,
              "lifting_independent_parameters": sum(p.numel() for p in model.frequency.lifting.parameters())}
    if hasattr(model, "spatial"):
        result["spatial_feature_shape"] = list(features.shape)
    del model
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODEL_NAMES, "both"), default="both")
    parser.add_argument("--output", default="docs/pretrained_verification.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(2)
    results = []
    for name in MODEL_NAMES if args.model == "both" else (args.model,):
        results.append(verify(name))
        print(f"OK: {name}, pesos públicos exactos, dimensiones y lifting Haar verificados", flush=True)
    save_json(resolve(args.output), results)


if __name__ == "__main__":
    main()
