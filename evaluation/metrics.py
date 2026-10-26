import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve


def binary_metrics(labels, probabilities, threshold=0.5):
    labels, probabilities = np.asarray(labels).reshape(-1), np.asarray(probabilities).reshape(-1)
    if not len(labels) or len(labels) != len(probabilities) or not np.isfinite(probabilities).all():
        raise ValueError("Probabilidades ausentes o no finitas")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("ROC-AUC requiere reales y falsos; no se sustituye por un valor inventado")
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(labels, probabilities)
    return {"accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(labels, probabilities)),
            "specificity": float(tn / (tn + fp)), "threshold": float(threshold),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "fpr": fpr.tolist(), "tpr": tpr.tolist(), "n": len(labels)}


def per_generator(labels, probabilities, generators, expected, threshold=0.5):
    labels, probabilities, generators = np.asarray(labels), np.asarray(probabilities), np.asarray(generators)
    results = {}
    for name in expected:
        mask = (labels == 0) | (generators == name)
        if not np.any((labels == 1) & (generators == name)):
            raise ValueError(f"No hay imágenes falsas de {name}")
        results[name] = binary_metrics(labels[mask], probabilities[mask], threshold)
    return results


def select_validation_threshold(labels, probabilities):
    y, p = np.asarray(labels), np.asarray(probabilities)
    order = np.argsort(-p, kind="stable")
    sorted_y, sorted_p = y[order], p[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_p)), len(sorted_p) - 1]
    tp = np.cumsum(sorted_y)[ends]
    f1 = 2 * tp / np.maximum(1, ends + 1 + y.sum())
    best = np.flatnonzero(f1 == f1.max())
    candidates = sorted_p[ends[best]]
    result = float(candidates[np.argmin(np.abs(candidates - 0.5))])
    if f1_score(y, p >= 0.5, zero_division=0) >= f1.max():
        result = 0.5
    return result
