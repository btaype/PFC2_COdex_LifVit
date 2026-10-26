import argparse
import json

import torch

from data.augmentation import build_transform
from data.preprocessing import check_prepared_face, load_rgb
from models import build_model
from models.weights import file_hash
from training.checkpoints import load_checkpoint
from utils.config import MODEL_NAMES, resolve


def main():
    parser = argparse.ArgumentParser(description="Predicción sobre un rostro ya recortado; salida Fake=1")
    parser.add_argument("--model", choices=MODEL_NAMES)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--threshold-file")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    state = load_checkpoint(resolve(args.checkpoint), args.model)
    cfg = state["training_config"]
    threshold = 0.5
    if args.threshold_file:
        selected = json.loads(resolve(args.threshold_file).read_text(encoding="utf-8"))
        if selected["split"] != "val" or selected["checkpoint_sha256"] != file_hash(resolve(args.checkpoint)) or selected["manifest_sha256"] != state["dataset_manifest_hash"]:
            parser.error("Umbral de otro checkpoint/validación")
        threshold = selected["threshold"]
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    model = build_model(cfg, initialize=False)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.to(device).eval()
    face = load_rgb(resolve(args.image))
    check_prepared_face(face)
    image = build_transform(cfg)(face).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(image)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Predicción no finita")
        probability = logits.sigmoid().item()
    print(json.dumps({"model": cfg["model"], "probability_fake": probability, "threshold": threshold, "prediction": "fake" if probability >= threshold else "real"}, indent=2))


if __name__ == "__main__":
    main()
