import torch
import torch.nn as nn
from einops import einsum
from jaxtyping import Float
from torch import Tensor


class SwiGLU(nn.Module):

    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        W1 = torch.zeros((d_ff, d_model), dtype=dtype, device=device)
        W2 = torch.zeros((d_model, d_ff), dtype=dtype, device=device)
        W3 = torch.zeros((d_ff, d_model), dtype=dtype, device=device)
        self.W1 = nn.Parameter(W1, requires_grad=True)
        self.W2 = nn.Parameter(W2, requires_grad=True)
        self.W3 = nn.Parameter(W3, requires_grad=True)

    def load_weights(self, weights: dict[str, Tensor]):
        self.load_state_dict(weights, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = einsum(self.W1, x, "d_ff d_model, ... d_model -> ... d_ff")
        silu = up * torch.sigmoid(up)
        glu = silu * einsum(self.W3, x, "d_ff d_model, ... d_model -> ... d_ff")
        swiglu = einsum(self.W2, glu, "d_model d_ff, ... d_ff -> ... d_model")
        return swiglu
