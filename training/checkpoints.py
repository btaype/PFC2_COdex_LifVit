import os
from pathlib import Path

import torch


def save_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, model_name=None):
    # Sin mmap: en Windows una vista viva impide reemplazar last.pth al reanudar.
    state = torch.load(path, map_location="cpu", weights_only=True)
    required = {"epoch", "model_state_dict", "model_name", "training_config", "preprocessing_config"}
    if not isinstance(state, dict) or required - state.keys():
        raise ValueError("No es un checkpoint de entrenamiento de este proyecto")
    if state["model_name"] not in ("liftvit", "liftvit_h") or (model_name and state["model_name"] != model_name):
        raise ValueError("El checkpoint pertenece a otra arquitectura")
    return state
