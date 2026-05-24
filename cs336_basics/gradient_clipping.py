from collections.abc import Iterable
import torch

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    eps = 1e-6
    grads = []
    for p in parameters:
        if p.grad is not None:
            grads.append(p.grad.data)
    l2_norm = torch.cat(grads).norm()
    if l2_norm >= max_l2_norm:
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.data *= max_l2_norm / (l2_norm + eps)
