import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Compara evaluaciones de ambos modelos sobre el mismo manifiesto")
    parser.add_argument("liftvit")
    parser.add_argument("liftvit_h")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    a, b = [json.loads(Path(p).read_text(encoding="utf-8")) for p in (args.liftvit, args.liftvit_h)]
    if a["model"] != "liftvit" or b["model"] != "liftvit_h":
        parser.error("Proporciona un resultado de cada arquitectura en el orden indicado")
    if a["evaluated_rows_hash"] != b["evaluated_rows_hash"] or a["split"] != b["split"] or a["protocol"] != b["protocol"]:
        parser.error("Las evaluaciones deben usar las mismas imágenes, partición y protocolo de entrenamiento")
    names = list(a["by_generator"])
    if set(names) != set(b["by_generator"]):
        parser.error("Los generadores evaluados difieren")
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.18, [a["by_generator"][n]["roc_auc"] for n in names], width=0.36, label="LiftViT")
    ax.bar(x + 0.18, [b["by_generator"][n]["roc_auc"] for n in names], width=0.36, label="LiftViT-H")
    ax.set(xticks=x, xticklabels=names, ylim=(0, 1), ylabel="ROC-AUC")
    ax.legend()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
