"""Solo dos detectores públicos: LiftViT y LiftViTH."""
from .liftvit import LiftViT
from .liftvit_h import LiftViTH


def build_model(config, initialize=True):
    classes = {"liftvit": LiftViT, "liftvit_h": LiftViTH}
    if config["model"] not in classes:
        raise ValueError("Solo se admiten liftvit y liftvit_h")
    return classes[config["model"]](config, initialize=initialize)
