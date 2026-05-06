import math


def schedule_lambda(progress: float, gamma: float = 10.0, max_lam: float = 1.0) -> float:
    """Standard DANN ramp."""
    progress = max(0.0, min(1.0, progress))
    return float(max_lam * (2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0))
