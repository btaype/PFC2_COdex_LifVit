import torch
from torch import nn

from .classifiers import classifier
from .cross_attention import BidirectionalCrossAttention
from .frequency_branch import FrequencyBranch
from .spatial_branch import SpatialBranch


class LiftViTH(nn.Module):
    def __init__(self, config, initialize=True):
        super().__init__()
        self.frequency = FrequencyBranch(config, initialize)
        self.spatial = SpatialBranch(config, initialize)
        self.cross_attention = BidirectionalCrossAttention()
        self.classifier = classifier(1536, 0.3)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (3, 224, 224):
            raise ValueError("La entrada debe ser [B,3,224,224]")
        f, s = self.cross_attention(self.frequency(x), self.spatial(x))
        return self.classifier(torch.cat((f[:, 0], s.mean(dim=1)), dim=1))
