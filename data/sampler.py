import math

import torch
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, seed=42):
        if batch_size % 2:
            raise ValueError("Batch equilibrado requiere tamaño par")
        self.indices = [[i for i, label in enumerate(labels) if int(label) == c] for c in (0, 1)]
        if any(not indices for indices in self.indices):
            raise ValueError("Se requieren imágenes de las dos clases")
        self.half = batch_size // 2
        self.seed, self.epoch = seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return math.ceil(max(map(len, self.indices)) / self.half)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        count = len(self) * self.half
        pools = []
        for indices in self.indices:
            pool = []
            while len(pool) < count:
                pool.extend(indices[i] for i in torch.randperm(len(indices), generator=generator).tolist())
            pools.append(pool[:count])
        for start in range(0, count, self.half):
            batch = pools[0][start:start + self.half] + pools[1][start:start + self.half]
            yield [batch[i] for i in torch.randperm(len(batch), generator=generator).tolist()]
