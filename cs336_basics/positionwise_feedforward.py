import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from .linear import Linear


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):

    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        self.w1 = Linear(d_model, d_ff, dtype=dtype, device=device)
        self.w2 = Linear(d_ff, d_model, dtype=dtype, device=device)
        self.w3 = Linear(d_model, d_ff, dtype=dtype, device=device)

    def load_weight(self, w1_weight: Float[Tensor, " d_ff d_model"], w2_weight: Float[Tensor, " d_model d_ff"], w3_weight: Float[Tensor, " d_ff d_model"],):
        weight = {"w1.weight": w1_weight, "w2.weight": w2_weight, "w3.weight": w3_weight}
        self.load_state_dict(weight, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = self.w1(x)
        silu_val = silu(up)
        glu = silu_val * self.w3(x)
        swiglu = self.w2(glu)
        return swiglu
