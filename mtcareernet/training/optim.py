import torch


def build_optimizer(model, name: str = "adamw", lr: float = 3e-4,
                    weight_decay: float = 1e-5, betas=(0.9, 0.999)):
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(name)
