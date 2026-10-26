import math

from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(optimizer, total_steps, warmup_fraction=0.05, min_lr_factor=0.05):
    warmup = max(1, math.ceil(total_steps * warmup_fraction))

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return min_lr_factor + (1.0 - min_lr_factor) * (1.0 + math.cos(math.pi * progress)) / 2

    return LambdaLR(optimizer, factor)
