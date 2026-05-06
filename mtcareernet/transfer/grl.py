import torch
from torch.autograd import Function


class _GRL(Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = float(lam)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


def grad_reverse(x, lam: float = 1.0):
    return _GRL.apply(x, lam)


class GradientReversalLayer(torch.nn.Module):
    def __init__(self, lam: float = 1.0):
        super().__init__()
        self.lam = lam

    def forward(self, x):
        return grad_reverse(x, self.lam)
