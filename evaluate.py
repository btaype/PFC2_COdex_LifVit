import argparse
import copy
import csv
import hashlib
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import torch
from torch.utils.data import DataLoader

from data.dataset import FaceDataset
from data.splits import group_audit, load_records
from evaluation.evaluate import collect_predictions
from evaluation.generalization import generalization_report
from evaluation.metrics import binary_metrics, per_generator, select_validation_threshold
from models import build_model
from models.weights import file_hash
from training.checkpoints import load_checkpoint
from utils.config import MODEL_NAMES, resolve
from utils.reproducibility import save_json, seed_everything
from visualization.plot_training import plot_evaluation


def main():
    parser = argparse.ArgumentParser(description="Evalúa el checkpoint exacto seleccionado")
    parser.add_argument("--model", choices=MODEL_NAMES)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--data-root", help="Dataset externo para evaluación entre conjuntos")
    parser.add_argument("--manifest")
    parser.add_argument("--generators", nargs="+")
    parser.add_argument("--threshold-file", help="Umbral seleccionado previamente en validación con este checkpoint")
    parser.add_argument("--select-threshold", action="store_true", help="Solo en val: maximizar F1 de Fake")
    parser.add_argument("--output")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.select_threshold and (args.split != "val" or args.threshold_file):
        parser.error("El umbral se selecciona exclusivamente en val, sin threshold-file")
    if bool(args.data_root) != bool(args.manifest):
        parser.error("Usa --data-root y --manifest juntos")
    state = load_checkpoint(resolve(args.checkpoint), args.model)
    cfg = state["training_config"]
    seed_everything(cfg["seed"])
    root, manifest = resolve(args.data_root or cfg["data"]["root"]), resolve(args.manifest or cfg["data"]["manifest"])
    data_config = copy.deepcopy(cfg["data"])
    data_config.update(root=str(root), manifest=str(manifest))
    if args.data_root:
        data_config["metadata_path"] = None
    rows = load_records(data_config, create_if_missing=bool(args.data_root) and not args.select_threshold)
    if args.select_threshold and (file_hash(manifest) != state["dataset_manifest_hash"] or args.generators and set(args.generators) != set(cfg["data"]["validation_generators"])):
        parser.error("Selecciona el umbral en la validación original del entrenamiento")
    names = args.generators or cfg["data"]["validation_generators" if args.split == "val" else "test_generators"]
    rows = [r for r in rows if r["split"] == args.split and (r["label"] == 0 or r["generator"] in names)]
    threshold = 0.5
    checkpoint_hash = file_hash(resolve(args.checkpoint))
    if args.threshold_file:
        selected = json.loads(resolve(args.threshold_file).read_text(encoding="utf-8"))
        if selected["split"] != "val" or selected["checkpoint_sha256"] != checkpoint_hash or selected["manifest_sha256"] != state["dataset_manifest_hash"]:
            parser.error("Umbral de otra validación o checkpoint")
        threshold = selected["threshold"]
    if not 0 <= threshold <= 1:
        parser.error("Umbral fuera de [0,1]")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    model = build_model(cfg, initialize=False)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.to(device)
    loader = DataLoader(FaceDataset(rows, root, cfg, args.split), batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"])
    pred = collect_predictions(model, loader, device)
    if args.select_threshold:
        threshold = select_validation_threshold(pred["labels"], pred["probabilities"])
    metrics = per_generator(pred["labels"], pred["probabilities"], pred["generators"], names, threshold)
    output = resolve(args.output or f"results/evaluation/{cfg['model']}_{datetime.now(ZoneInfo('America/Lima')).strftime('%Y%m%d_%H%M%S_%f')}")
    output.mkdir(parents=True, exist_ok=False)
    row_hash = hashlib.sha256(json.dumps(sorted((r["path"], r["sha256"], r["label"], r["generator"]) for r in rows)).encode()).hexdigest()
    report = {"model": cfg["model"], "protocol": cfg["protocol"], "checkpoint": str(resolve(args.checkpoint)), "checkpoint_sha256": checkpoint_hash,
              "manifest_sha256": file_hash(manifest), "evaluated_rows_hash": row_hash, "split": args.split, "threshold": threshold,
              "data_audit": group_audit(rows),
              "by_generator": metrics, "macro_auc": sum(v["roc_auc"] for v in metrics.values()) / len(metrics),
              "pooled": binary_metrics(pred["labels"], pred["probabilities"], threshold),
              "generalization": generalization_report(metrics) if cfg["protocol"] == "diffusion_training" else None}
    save_json(output / "metrics.json", report)
    if args.select_threshold:
        save_json(output / "threshold.json", {"threshold": threshold, "criterion": "maximum_validation_fake_f1", "split": "val", "checkpoint_sha256": checkpoint_hash, "manifest_sha256": file_hash(manifest)})
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "label", "generator", "probability_fake"])
        writer.writerows(zip(pred["paths"], pred["labels"], pred["generators"], pred["probabilities"]))
    plot_evaluation(metrics, output)
    print(f"AUC macro={report['macro_auc']:.5f}; resultados: {output}")


if __name__ == "__main__":
    main()
