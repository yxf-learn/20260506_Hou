import math


def warmup_cosine(step: int, total: int, warmup: int = 1000, base_lr: float = 3e-4) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))


class ExpDecaySchedule:
    def __init__(self, base_lr: float = 3e-4, gamma: float = 0.97, every: int = 1):
        self.base_lr = base_lr; self.gamma = gamma; self.every = every

    def __call__(self, epoch: int) -> float:
        return self.base_lr * (self.gamma ** (epoch // self.every))
