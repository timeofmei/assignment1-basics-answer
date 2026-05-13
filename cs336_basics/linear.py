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

        weight = torch.zeros((out_features, in_features), dtype=dtype, device=device)
        mean = 0
        std = np.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(weight, mean=mean, std=std, a=-3*std, b=3*std)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def load_weight(self, weight: Float[Tensor, " d_out d_in"]):
        self.load_state_dict({"weight": weight}, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")
