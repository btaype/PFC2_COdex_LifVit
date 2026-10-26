"""Prueba técnica con ruido aleatorio: sus pérdidas no son resultados de detección."""
import argparse
import gc

import torch

from models import build_model
from training.optimizer import build_optimizer, set_training_stage
from utils.config import MODEL_NAMES, load_config
from utils.reproducibility import save_json, seed_everything


def synthetic_step(name, device, stage=6, batch_size=1, amp=False):
    cfg = load_config(model=name)
    seed_everything(cfg["seed"])
    model = build_model(cfg, initialize=False).to(device).train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    set_training_stage(model, stage)
    optimizer = build_optimizer(model, cfg)
    initial = {k: p.detach().clone() for k, p in model.frequency.lifting.named_parameters()}
    shapes = {}
    handles = []
    def capture(key):
        def hook(module, inputs, output):
            shapes[key] = list(output.shape)
        return hook
    handles.append(model.frequency.lifting.register_forward_hook(capture("lifting")))
    handles.append(model.frequency.patch_embedding.register_forward_hook(capture("patch_embedding_grid")))
    handles.append(model.frequency.register_forward_hook(capture("frequency_tokens")))
    if hasattr(model, "spatial"):
        handles.append(model.spatial.register_forward_hook(capture("spatial_tokens")))
    x = torch.rand(batch_size, 3, 224, 224, device=device)
    labels = (torch.arange(batch_size, device=device) % 2).float().unsqueeze(1)
    amp = amp and device.type == "cuda"
    dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp and dtype == torch.float16)
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
        logits = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    for handle in handles:
        handle.remove()
    assert logits.shape == (batch_size, 1)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = {k: float(p.grad.norm()) for k, p in model.frequency.lifting.named_parameters() if p.grad is not None}
    assert len(gradients) == 4 and all(v > 0 for v in gradients.values()), gradients
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), "Gradientes NaN/Inf"
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    changes = {k: float((p.detach() - initial[k]).norm()) for k, p in model.frequency.lifting.named_parameters()}
    assert all(v > 0 for v in changes.values()), changes
    for name_, kernel in model.frequency.lifting.effective_kernels().items():
        target = 1.0 if name_.startswith("P") else 0.5
        assert torch.allclose(kernel.sum((1, 2, 3)), torch.full((3,), target, device=device), atol=1e-6, rtol=0)
    result = {"model": name, "synthetic_only": True, "device": str(device), "stage_epoch": stage,
              "backbone_names": {"vit": cfg["vit"]["model_name"], "efficientnet": cfg["efficientnet"]["model_name"] if name == "liftvit_h" else None},
              "pretrained_weights_loaded": False,
              "test_batch_size": batch_size, "logits_shape": list(logits.shape), "intermediate_shapes": shapes,
              "precision": str(dtype) if amp else "torch.float32", "gpu_peak_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
              "loss": float(loss.detach()), "lifting_gradient_norms": gradients, "lifting_parameter_changes": changes,
              "total_parameters": sum(p.numel() for p in model.parameters())}
    del model, optimizer, logits, loss
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODEL_NAMES, "both"), default="both")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--stage", type=int, choices=(1, 3, 6), default=6)
    parser.add_argument("--batch-size", type=int, default=1, help="Solo para esta prueba técnica; no cambia el batch de entrenamiento")
    parser.add_argument("--physical-batch", action="store_true", help="Probar el batch físico prescrito: 10 o 4, según el modelo")
    parser.add_argument("--amp", action="store_true", help="Usar BF16/FP16 según soporte de CUDA")
    parser.add_argument("--output", default="docs/smoke_test.json")
    args = parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    names = MODEL_NAMES if args.model == "both" else (args.model,)
    results = []
    for name in names:
        batch = (10 if name == "liftvit" else 4) if args.physical_batch else args.batch_size
        result = synthetic_step(name, device, args.stage, batch, args.amp)
        results.append(result)
        print(f"OK: {name}, forward/backward/AdamW, batch de prueba={batch}, dispositivo={device}", flush=True)
    from utils.config import resolve
    save_json(resolve(args.output), results)


if __name__ == "__main__":
    main()
