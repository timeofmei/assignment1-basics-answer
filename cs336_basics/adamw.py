from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["m"] = torch.zeros(p.shape)
                state["v"] = torch.zeros(p.shape)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta_1 = group["betas"][0]
            beta_2 = group["betas"][1]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)

                grad = p.grad.data
                lr_t = lr / (1 - beta_1 ** t) * math.sqrt(1 - beta_2 ** t)
                p.data -= lr * weight_decay * p.data
                state["m"] = beta_1 * state["m"] + (1 - beta_1) * grad
                state["v"] = beta_2 * state["v"] + (1 - beta_2) * grad ** 2
                p.data -= lr_t / (state["v"].sqrt() + eps) * state["m"]

                state["t"] = t + 1


        return loss