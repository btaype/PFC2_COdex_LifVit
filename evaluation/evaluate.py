import torch


@torch.inference_mode()
def collect_predictions(model, loader, device, criterion=None):
    model.eval()
    labels, probabilities, generators, paths = [], [], [], []
    total_loss, count = 0.0, 0
    for batch in loader:
        x, y = batch["image"].to(device), batch["label"].to(device)
        logits = model(x)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Logits no finitos en evaluación")
        if criterion is not None:
            total_loss += criterion(logits, y).item() * len(y)
        count += len(y)
        labels.extend(y.flatten().cpu().tolist())
        probabilities.extend(logits.sigmoid().flatten().cpu().tolist())
        generators.extend(batch["generator"])
        paths.extend(batch["path"])
    return {"labels": labels, "probabilities": probabilities, "generators": generators, "paths": paths,
            "loss": total_loss / count if count and criterion is not None else None}
