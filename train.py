import argparse
import json

from utils.config import MODEL_NAMES, load_config, resolve


def main():
    parser = argparse.ArgumentParser(description="Entrena exclusivamente LiftViT o LiftViT-H")
    parser.add_argument("--model", choices=MODEL_NAMES)
    parser.add_argument("--config", help="YAML del modelo/protocolo")
    parser.add_argument("--resume", help="Checkpoint exacto; recupera su configuración")
    parser.add_argument("--run-name")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--allow-random-init", action="store_true", help="Permitir explícitamente entrenamiento desde cero")
    parser.add_argument("--dry-run", action="store_true", help="Muestra plan y carpetas; no necesita imágenes ni descarga pesos")
    args = parser.parse_args()
    if args.resume and not args.config:
        from training.checkpoints import load_checkpoint
        cfg = load_checkpoint(resolve(args.resume), args.model)["training_config"]
    else:
        cfg = load_config(args.config, args.model)
    if args.allow_random_init:
        cfg["training"]["allow_random_init"] = True
    if args.device:
        cfg["training"]["device"] = args.device
    if args.dry_run:
        print(json.dumps({"model": cfg["model"], "protocol": cfg["protocol"],
                          "data_root": str(resolve(cfg["data"]["root"])), "manifest": str(resolve(cfg["data"]["manifest"])),
                          "manifest_exists": resolve(cfg["data"]["manifest"]).is_file(),
                          "auto_manifest": cfg["data"].get("auto_manifest", True),
                          "metadata_path": cfg["data"].get("metadata_path"),
                          "require_group_ids": cfg["data"].get("require_group_ids", False),
                          "physical_batch": cfg["training"]["batch_size"],
                          "accumulation": cfg["training"]["gradient_accumulation_steps"],
                          "nominal_effective_batch": cfg["training"]["batch_size"] * cfg["training"]["gradient_accumulation_steps"],
                          "epochs": cfg["training"]["epochs"], "vit": cfg["vit"],
                          "efficientnet": cfg["efficientnet"] if cfg["model"] == "liftvit_h" else None}, indent=2))
        return
    from training.trainer import train
    try:
        run = train(cfg, resolve(args.resume) if args.resume else None, args.run_name)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"No se inició el entrenamiento: {exc}\n")
    print(f"Checkpoints: {run}")


if __name__ == "__main__":
    main()
