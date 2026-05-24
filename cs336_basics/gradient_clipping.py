from collections.abc import Iterable
import torch

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    eps = 1e-6
    l2_norm = 0
    for p in parameters:
        if p.grad is not None:
            l2_norm += p.grad.data.pow(2).sum()
    l2_norm = l2_norm ** 0.5
    if l2_norm >= max_l2_norm:
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.data *= max_l2_norm / (l2_norm + eps)
