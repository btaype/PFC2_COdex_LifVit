"""Prueba integral temporal con las arquitecturas reales y datos sintéticos; no mide detección."""
import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from models.weights import file_hash
from data.splits import read_manifest
from training.checkpoints import load_checkpoint
from training.trainer import train
from utils.config import MODEL_NAMES, ROOT, load_config, resolve
from utils.reproducibility import save_json


def verify(name):
    from PIL import Image
    with tempfile.TemporaryDirectory(prefix=f"liftvit_pipeline_{name}_") as tmp:
        folder = Path(tmp)
        root, manifest = folder / "dataset", folder / "manifest.csv"
        rng = np.random.default_rng(42)
        for split in ("train", "val", "test"):
            categories = ["real"] * (5 if split == "train" else 2) + ["ddpm", "ldm"] * (3 if split == "train" else 1)
            for index, generator in enumerate(categories):
                relative = Path(split) / ("real" if generator == "real" else "fake/" + generator) / f"sample_{index}.png"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)).save(path)
        cfg = load_config(model=name)
        cfg["data"].update(root=str(root), manifest=str(manifest))
        # Esta prueba verifica la ruta por carpetas, sin CSV de metadatos requerido.
        cfg["vit"]["pretrained"] = False
        cfg["efficientnet"]["pretrained"] = False
        cfg["output_root"] = str(folder / "artifacts")
        cfg["training"].update(device="cpu", epochs=2, checkpoint_interval=1, tensorboard=False, allow_random_init=True)
        original_run = train(cfg, run_name="synthetic_original")
        original = load_checkpoint(original_run / "last.pth", name)
        assert original["data_audit"]["images_with_group_id"] == 0
        assert all(row["group_id"] == "" for row in read_manifest(manifest, root))
        original_state = {k: v.clone() for k, v in original["model_state_dict"].items()}
        expected_step = original["global_step"]
        del original
        resumed_run = train(cfg, resume=original_run / "epoch_001.pth", run_name="synthetic_resumed")
        resumed = load_checkpoint(resumed_run / "last.pth", name)
        assert original_run != resumed_run
        assert resumed["global_step"] == expected_step
        for key, value in resumed["model_state_dict"].items():
            torch.testing.assert_close(value, original_state[key], atol=0, rtol=0, msg=lambda msg: f"Resume difiere en {key}: {msg}")
        del resumed, original_state
        gc.collect()
        checkpoint = resumed_run / "last.pth"
        threshold_dir = folder / "threshold"
        child_env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        def run_cli(arguments):
            result = subprocess.run([sys.executable, *arguments], cwd=ROOT, env=child_env, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            return result.stdout
        checkpoint_hash = file_hash(checkpoint)
        run_cli(["evaluate.py", "--checkpoint", str(checkpoint), "--split", "val", "--select-threshold", "--output", str(threshold_dir), "--device", "cpu"])
        eval_dir = folder / "evaluation"
        run_cli(["evaluate.py", "--checkpoint", str(checkpoint), "--generators", "ddpm", "ldm", "--threshold-file", str(threshold_dir / "threshold.json"), "--output", str(eval_dir), "--device", "cpu"])
        prediction = run_cli(["predict.py", "--checkpoint", str(checkpoint), "--image", str(root / "test/real/sample_0.png"), "--threshold-file", str(threshold_dir / "threshold.json"), "--device", "cpu"])
        assert file_hash(checkpoint) == checkpoint_hash
        assert 0 <= json.loads(prediction)["probability_fake"] <= 1
        assert (eval_dir / "roc.png").is_file() and (eval_dir / "predictions.csv").is_file()
        report = {"model": name, "synthetic_only": True, "training_epochs": 2, "physical_batch": cfg["training"]["batch_size"],
                  "resume_from_epoch": 1, "resumed_parameters_bitwise_equal_cpu": True, "global_optimizer_steps": expected_step,
                  "manifest_validation": "passed", "folder_only_without_metadata": True, "automatic_manifest": True,
                  "invented_group_ids": False, "evaluation_cli": "passed", "validation_threshold_cli": "passed", "prediction_cli": "passed"}
        logging_handlers = __import__("logging").getLogger().handlers[:]
        for handler in logging_handlers:
            handler.close()
            __import__("logging").getLogger().removeHandler(handler)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODEL_NAMES, "both"), default="both")
    parser.add_argument("--output", default="docs/pipeline_test.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    results = []
    for name in MODEL_NAMES if args.model == "both" else (args.model,):
        results.append(verify(name))
        print(f"OK integral: {name}; solo datos sintéticos; archivos temporales eliminados", flush=True)
    save_json(resolve(args.output), results)


if __name__ == "__main__":
    main()
