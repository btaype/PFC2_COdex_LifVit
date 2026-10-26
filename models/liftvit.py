from torch import nn

from .classifiers import classifier
from .frequency_branch import FrequencyBranch


class LiftViT(nn.Module):
    def __init__(self, config, initialize=True):
        super().__init__()
        self.frequency = FrequencyBranch(config, initialize)
        self.classifier = classifier(768, 0.2)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (3, 224, 224):
            raise ValueError("La entrada debe ser [B,3,224,224]")
        return self.classifier(self.frequency(x)[:, 0])
