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


def plot_subbands(model, image, output):
    with torch.inference_mode():
        branch = model.frequency
        bands = branch.lifting.subbands((image - branch.mean) / branch.std)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax, name, band in zip(axes, ("LL", "LH", "HL", "HH"), bands):
        values = band[0].mean(0).cpu().numpy()
        ax.imshow(values, cmap="gray")
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle("Promedio RGB de cada subbanda; escalas independientes para visualización")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    state = load_checkpoint(args.checkpoint)
    cfg = state["training_config"]
    model = build_model(cfg, initialize=False).eval()
    model.load_state_dict(state["model_state_dict"])
    image = build_transform(cfg)(load_rgb(args.image)).unsqueeze(0)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plot_subbands(model, image, args.output)


if __name__ == "__main__":
    main()
