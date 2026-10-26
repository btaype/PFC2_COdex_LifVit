from torch import nn


def classifier(input_dim, dropout):
    return nn.Sequential(nn.Linear(input_dim, 512), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(512, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1))
