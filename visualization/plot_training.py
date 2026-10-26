import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_history(history, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = [r["epoch"] for r in history]
    for name in ("train_loss", "validation_loss"):
        axes[0].plot(epochs, [r[name] for r in history], label=name)
    axes[0].legend()
    axes[0].set_title("BCE")
    axes[1].plot(epochs, [r["validation_macro_auc"] for r in history])
    axes[1].set_title("AUC macro de validación")
    axes[2].plot(epochs, [r["lr"] for r in history])
    axes[2].set_title("Learning rate (grupo lifting)")
    for ax in axes:
        ax.set_xlabel("Época")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "training.png", dpi=160)
    plt.close(fig)


def plot_evaluation(metrics, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, values in metrics.items():
        ax.plot(values["fpr"], values["tpr"], label=f"{name}: AUC={values['roc_auc']:.3f}")
        matrix_fig, matrix_ax = plt.subplots(figsize=(4, 4))
        matrix_ax.imshow(values["confusion_matrix"], cmap="Blues")
        for i in range(2):
            for j in range(2):
                matrix_ax.text(j, i, str(values["confusion_matrix"][i][j]), ha="center", va="center")
        matrix_ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"], xlabel="Predicción", ylabel="Verdad", title=name)
        matrix_fig.tight_layout()
        matrix_fig.savefig(output / f"confusion_{name}.png", dpi=160)
        plt.close(matrix_fig)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="Falsos positivos", ylabel="Verdaderos positivos", title="ROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "roc.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Grafica un historial de entrenamiento existente")
    parser.add_argument("history")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot_history(json.loads(Path(args.history).read_text(encoding="utf-8")), args.output)


if __name__ == "__main__":
    main()
