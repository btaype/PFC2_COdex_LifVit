from torch import nn


def build_loss():
    # Batches equilibrados: sin ponderación adicional ni sigmoid previo.
    return nn.BCEWithLogitsLoss()
