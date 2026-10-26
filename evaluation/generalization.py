def generalization_report(metrics):
    if not {"ddpm", "ldm", "ddim", "sdxl"}.issubset(metrics):
        return None
    seen = (metrics["ddpm"]["roc_auc"] + metrics["ldm"]["roc_auc"]) / 2
    unseen = (metrics["ddim"]["roc_auc"] + metrics["sdxl"]["roc_auc"]) / 2
    return {"auc_seen": seen, "auc_unseen": unseen, "delta_auc": seen - unseen,
            "interpretation": "Etiquetas visto/no visto sujetas a auditoría del checkpoint generativo y sampler; delta pequeño no demuestra AUC alto."}
