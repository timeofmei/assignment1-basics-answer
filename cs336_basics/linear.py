import torch
import torch.nn as nn
import numpy as np
from einops import einsum
from jaxtyping import Float
from torch import Tensor


class Linear(nn.Module):

    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        W = torch.zeros((out_features, in_features), dtype=dtype, device=device)
        mean = 0
        std = np.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(W, mean=mean, std=std, a=-3*std, b=3*std)
        self.W = nn.Parameter(W, requires_grad=True)

    def load_weights(self, weights: Float[Tensor, " d_out d_in"]):
        self.load_state_dict({"W": weights}, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.W, x, "d_out d_in, ... d_in -> ... d_out")
