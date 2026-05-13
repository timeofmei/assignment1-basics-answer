import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        weight = torch.zeros(d_model, dtype=dtype, device=device)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def load_weight(self, weight: Float[Tensor, " d_model"]):
        self.load_state_dict({"weight": weight}, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(self.eps + x.pow(2).sum(dim=-1, keepdim=True) / self.d_model)
        result = x / rms * self.weight
        return result.to(in_dtype)
