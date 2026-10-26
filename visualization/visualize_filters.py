import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def plot_filters(initial, final, output):
    fig, axes = plt.subplots(4, 3, figsize=(9, 10))
    for row, name in enumerate(("P_f", "U_f", "P_c", "U_c")):
        for col, channel in enumerate("RGB"):
            delta = (final[name] - initial[name])[col, 0].detach().cpu().numpy()
            limit = max(float(abs(delta).max()), 1e-8)
            axes[row, col].imshow(delta, cmap="coolwarm", vmin=-limit, vmax=limit)
            axes[row, col].set_title(f"{name}/{channel}: final - inicial")
            for i in range(3):
                for j in range(3):
                    axes[row, col].text(j, i, f"{delta[i,j]:.3g}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kernels", help="lifting_kernels.pth")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    weights = torch.load(args.kernels, map_location="cpu", weights_only=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plot_filters(weights["initial"], weights["final"], args.output)


if __name__ == "__main__":
    main()
