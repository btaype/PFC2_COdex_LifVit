import copy
import hashlib
import json
import logging
import math
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import FaceDataset
from data.sampler import BalancedBatchSampler
from data.splits import group_audit, load_records, validate_protocol
from evaluation.evaluate import collect_predictions
from evaluation.metrics import binary_metrics, per_generator
from models import build_model
from models.weights import file_hash
from utils.config import resolve
from utils.reproducibility import environment, restore_rng, rng_state, save_json, seed_everything, seed_worker
from .checkpoints import load_checkpoint, save_checkpoint
from .early_stopping import EarlyStopping
from .losses import build_loss
from .optimizer import build_optimizer, set_training_stage
from .scheduler import build_scheduler


def gradient_norm(parameters):
    values = [p.grad.detach().float().norm().square() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def lifting_diagnostics(model, initial):
    lifting = model.frequency.lifting
    result = {}
    for name, kernel in lifting.effective_kernels().items():
        values = kernel.detach().float().cpu()
        result[name + "_sum"] = values.sum((1, 2, 3)).tolist()
        result[name + "_change"] = float((values - initial[name]).norm())
        target = 1.0 if name.startswith("P") else 0.5
        if not torch.allclose(values.sum((1, 2, 3)), torch.full((3,), target), atol=1e-5, rtol=0):
            raise FloatingPointError("Los filtros incumplen las sumas del lifting")
    return result


def _new_run(cfg, run_name):
    folder = "LiftViT" if cfg["model"] == "liftvit" else "LiftViT_H"
    if not run_name:
        run_name = datetime.now(ZoneInfo("America/Lima")).strftime("%Y%m%d_%H%M%S_%f")
    if Path(run_name).name != run_name or run_name in (".", ".."):
        raise ValueError("run-name debe ser un nombre simple de carpeta")
    run = resolve(cfg["output_root"]) / "checkpoints" / folder / cfg["protocol"] / run_name
    run.mkdir(parents=True, exist_ok=False)
    return run


def train(cfg, resume=None, run_name=None):
    seed_everything(cfg["seed"])
    root, manifest = resolve(cfg["data"]["root"]), resolve(cfg["data"]["manifest"])
    records = load_records(cfg["data"], create_if_missing=resume is None)
    validate_protocol(records, cfg)
    audit = group_audit(records)
    dataset_hash = file_hash(manifest)
    split_hash = hashlib.sha256(json.dumps(sorted((r["path"], r["split"], r["group_id"]) for r in records)).encode()).hexdigest()
    previous = load_checkpoint(resume, cfg["model"]) if resume else None
    if previous:
        if cfg != previous["training_config"]:
            raise ValueError("La configuración difiere del checkpoint. Reanuda con su configuración original.")
        if dataset_hash != previous["dataset_manifest_hash"] or split_hash != previous["split_manifest_hash"]:
            raise ValueError("El dataset o las particiones cambiaron desde el checkpoint")
        run = Path(resume).resolve().parent
        last_path = run / "last.pth"
        if last_path.exists() and previous["epoch"] < load_checkpoint(last_path, cfg["model"])["epoch"]:
            run = _new_run(cfg, run_name)
    else:
        for key in ("vit", "efficientnet") if cfg["model"] == "liftvit_h" else ("vit",):
            if not cfg[key]["pretrained"] and not cfg[key].get("checkpoint_path") and not cfg["training"]["allow_random_init"]:
                raise ValueError(f"Elige pesos para {key}: checkpoint_path local o pretrained: true. Para una prueba deliberada usa --allow-random-init.")
        run = _new_run(cfg, run_name)
    artifact_root = resolve(cfg["output_root"])
    relative_run = run.relative_to(artifact_root / "checkpoints") if run.is_relative_to(artifact_root / "checkpoints") else Path(run.name)
    log_dir, result_dir = artifact_root / "logs" / relative_run, artifact_root / "results" / relative_run
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "training.log", encoding="utf-8")], force=True)
    runtime_cfg = copy.deepcopy(cfg)
    for key in ("vit", "efficientnet"):
        if runtime_cfg[key].get("checkpoint_path"):
            runtime_cfg[key]["checkpoint_path"] = str(resolve(runtime_cfg[key]["checkpoint_path"]))
    model = build_model(runtime_cfg, initialize=previous is None)
    selected_device = cfg["training"].get("device", "auto")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if selected_device == "auto" else selected_device)
    model.to(device)
    logging.info("Dispositivo=%s; batch físico=%s; acumulación=%s", device, cfg["training"]["batch_size"], cfg["training"]["gradient_accumulation_steps"])
    save_json(run / "config.json", cfg)
    save_json(run / "environment.json", environment())
    save_json(run / "data_audit.json", audit)
    if audit["images_without_group_id"]:
        logging.info("%d imágenes sin group_id: se verifican duplicados y los grupos conocidos; la independencia entre identidades queda sin comprobar.", audit["images_without_group_id"])
    sources = previous["pretrained_sources"] if previous else {"vit": model.frequency.source}
    if not previous and hasattr(model, "spatial"):
        sources["efficientnet"] = model.spatial.source
    save_json(run / "pretrained_sources.json", sources)
    train_set, val_set = FaceDataset(records, root, cfg, "train"), FaceDataset(records, root, cfg, "val")
    sampler = BalancedBatchSampler([r["label"] for r in train_set.records], cfg["training"]["batch_size"], cfg["seed"])
    loader_rng = torch.Generator().manual_seed(cfg["seed"])
    loader_options = {"num_workers": cfg["data"]["num_workers"], "pin_memory": device.type == "cuda", "worker_init_fn": seed_worker, "generator": loader_rng}
    train_loader = DataLoader(train_set, batch_sampler=sampler, **loader_options)
    val_loader = DataLoader(val_set, batch_size=cfg["training"]["batch_size"], shuffle=False, **loader_options)
    t = cfg["training"]
    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = math.ceil(len(train_loader) / t["gradient_accumulation_steps"])
    scheduler = build_scheduler(optimizer, t["epochs"] * steps_per_epoch, t["warmup_fraction"], t["min_lr_factor"])
    amp = t["amp"] and device.type == "cuda"
    dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp and dtype == torch.float16)
    logging.info("AMP=%s dtype=%s; pasos/época=%s", amp, dtype if amp else torch.float32, steps_per_epoch)
    stopper = EarlyStopping(**cfg["early_stopping"])
    criterion = build_loss()
    epoch_start, global_step, best_auc, best_epoch, history = 1, 0, -1.0, 0, []
    initial = {k: v.detach().cpu().clone() for k, v in model.frequency.lifting.effective_kernels().items()}
    if previous:
        model.load_state_dict(previous["model_state_dict"], strict=True)
        optimizer.load_state_dict(previous["optimizer_state_dict"])
        scheduler.load_state_dict(previous["scheduler_state_dict"])
        scaler.load_state_dict(previous["scaler_state_dict"])
        stopper.load_state_dict(previous["early_stopping_state_dict"])
        epoch_start, global_step = previous["epoch"] + 1, previous["global_step"]
        best_auc, best_epoch = previous["best_val_auc"], previous["best_epoch"]
        initial, history = previous["lifting_initial"], previous["history"]
        restore_rng(previous["rng_states"])
        loader_rng.set_state(previous["loader_rng_state"])
        if previous.get("stopped_early"):
            logging.info("El checkpoint ya alcanzó early stopping; no se reinicia su paciencia.")
            return run
        del previous
    writer = None
    if t["tensorboard"]:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(log_dir), purge_step=epoch_start if resume else None)
    try:
        for epoch in range(epoch_start, t["epochs"] + 1):
            start = time.perf_counter()
            sampler.set_epoch(epoch)
            model.train()
            set_training_stage(model, epoch)
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            loss_sum, count, correct, norms, lifting_norms = 0.0, 0, 0, [], []
            branches = {"vit": [], "efficientnet": [], "cross_attention": []}
            for i, batch in enumerate(tqdm(train_loader, desc=f"Época {epoch}/{t['epochs']}")):
                x, y = batch["image"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)
                group_start = (i // t["gradient_accumulation_steps"]) * t["gradient_accumulation_steps"]
                group_length = min(t["gradient_accumulation_steps"], len(train_loader) - group_start)
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                    logits = model(x)
                    loss = criterion(logits, y)
                if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                    raise FloatingPointError("Loss/logits NaN o Inf; se detiene antes de guardar pesos dañados")
                scaler.scale(loss / group_length).backward()
                loss_sum += float(loss.detach()) * len(y)
                count += len(y)
                correct += int(((logits.detach() >= 0) == (y >= 0.5)).sum())
                if (i + 1) % t["gradient_accumulation_steps"] == 0 or i + 1 == len(train_loader):
                    scaler.unscale_(optimizer)
                    lifting_norms.append(gradient_norm(model.frequency.lifting.parameters()))
                    branches["vit"].append(gradient_norm(model.frequency.vit.parameters()))
                    if hasattr(model, "spatial"):
                        branches["efficientnet"].append(gradient_norm(model.spatial.backbone.parameters()))
                        branches["cross_attention"].append(gradient_norm(model.cross_attention.parameters()))
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["max_grad_norm"], error_if_nonfinite=True)
                    norms.append(float(norm))
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    global_step += 1
                    optimizer.zero_grad(set_to_none=True)
            prediction = collect_predictions(model, val_loader, device, criterion)
            metrics = binary_metrics(prediction["labels"], prediction["probabilities"])
            grouped = per_generator(prediction["labels"], prediction["probabilities"], prediction["generators"], cfg["data"]["validation_generators"])
            val_auc = sum(m["roc_auc"] for m in grouped.values()) / len(grouped)
            diagnostics = lifting_diagnostics(model, initial)
            improved = val_auc > best_auc
            if improved:
                best_auc, best_epoch = val_auc, epoch
            stop = stopper.update(val_auc, epoch)
            row = {"epoch": epoch, "train_loss": loss_sum / count, "validation_loss": prediction["loss"],
                   "train_accuracy": correct / count, "validation_accuracy": metrics["accuracy"],
                   "validation_macro_auc": val_auc, "validation_f1": metrics["f1"], "best_validation_auc": best_auc,
                   "gradient_norm": sum(norms) / len(norms), "lifting_gradient_norm": sum(lifting_norms) / len(lifting_norms),
                   "gpu_peak_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
                   "epoch_seconds": time.perf_counter() - start, "lr": optimizer.param_groups[0]["lr"]}
            for key, values in branches.items():
                if values:
                    row[f"{key}_gradient_norm"] = sum(values) / len(values)
                    if row[f"{key}_gradient_norm"] == 0 and (key == "cross_attention" or epoch >= 3):
                        warnings.warn(f"La rama {key} no recibió gradientes")
            if not row["lifting_gradient_norm"]:
                warnings.warn("El lifting no recibió gradientes")
            for group in optimizer.param_groups:
                row["lr_" + group["name"].replace("/", "_")] = group["lr"]
            history.append(row)
            save_json(result_dir / "history.json", history)
            save_json(result_dir / f"lifting_epoch_{epoch:03}.json", diagnostics)
            save_json(result_dir / f"validation_epoch_{epoch:03}.json", grouped)
            if writer:
                for key, value in row.items():
                    writer.add_scalar(key, value, epoch)
                for key, values in diagnostics.items():
                    for c, value in enumerate(values if isinstance(values, list) else [values]):
                        writer.add_scalar(f"lifting/{key}/{c}", value, epoch)
            logging.info("Época=%d train_loss=%.5f val_loss=%.5f train_acc=%.4f val_acc=%.4f val_macro_auc=%.5f val_f1=%.4f lr=%.3g grad=%.3g GPU=%.2fGB best=%.5f tiempo=%.1fs", epoch, row["train_loss"], row["validation_loss"], row["train_accuracy"], row["validation_accuracy"], val_auc, row["validation_f1"], row["lr"], row["gradient_norm"], row["gpu_peak_bytes"] / 1024**3, best_auc, row["epoch_seconds"])
            state = {"epoch": epoch, "global_step": global_step, "model_state_dict": model.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                     "scaler_state_dict": scaler.state_dict(), "best_val_auc": best_auc, "best_epoch": best_epoch,
                     "training_config": cfg, "preprocessing_config": cfg["preprocessing"], "seed": cfg["seed"],
                     "rng_states": rng_state(), "loader_rng_state": loader_rng.get_state(),
                     "dataset_manifest_hash": dataset_hash, "split_manifest_hash": split_hash, "model_name": cfg["model"],
                     "early_stopping_state_dict": stopper.state_dict(), "history": history, "lifting_initial": initial,
                     "pretrained_sources": sources, "data_audit": audit, "stopped_early": stop}
            save_checkpoint(run / "last.pth", state)
            if improved:
                save_checkpoint(run / "best_auc.pth", state)
            if epoch % t["checkpoint_interval"] == 0:
                save_checkpoint(run / f"epoch_{epoch:03}.pth", state)
            if stop:
                logging.info("Early stopping en época %d; mejor época=%d", epoch, best_epoch)
                break
        final = {k: v.detach().cpu() for k, v in model.frequency.lifting.effective_kernels().items()}
        torch.save({"initial": initial, "final": final}, result_dir / "lifting_kernels.pth")
        if history:
            from visualization.plot_training import plot_history
            plot_history(history, result_dir)
        return run
    except torch.cuda.OutOfMemoryError:
        logging.exception("Memoria CUDA insuficiente. El batch físico no se modifica; usa una GPU con más memoria o revisa el consumo antes de continuar.")
        raise
    finally:
        if writer:
            writer.close()
