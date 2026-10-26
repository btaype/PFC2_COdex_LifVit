import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data.augmentation import build_transform
from data.preprocessing import load_rgb
from models import build_model
from training.checkpoints import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Atención cruzada de LiftViT-H; no equivale a explicación causal")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    state = load_checkpoint(args.checkpoint, "liftvit_h")
    cfg = state["training_config"]
    model = build_model(cfg, initialize=False).eval()
    model.load_state_dict(state["model_state_dict"])
    x = build_transform(cfg)(load_rgb(args.image)).unsqueeze(0)
    with torch.inference_mode():
        _, _, weights, _ = model.cross_attention(model.frequency(x), model.spatial(x), return_attention=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(weights[0, :, 0, :].mean(0).reshape(7, 7).cpu().numpy(), cmap="magma")
    ax.set_title("Consulta CLS → espacio; promedio de 12 cabezas")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
