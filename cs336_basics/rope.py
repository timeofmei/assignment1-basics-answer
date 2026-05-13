import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor


class RoPE(nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        R = []
        for i in range(max_seq_len):
            R_i = []
            for k in range(1, int(d_k / 2 + 1)):
                R_i.append(self._get_rik(i, k))
            R.append(torch.stack(R_i))
        R = torch.stack(R).to(device=device)
        self.register_buffer("R", R, persistent=False)

    def _get_rik(self, i: int, k: int):
        angle = i / np.pow(self.theta, (2 * k - 2) / self.d_k)
        cos = np.cos(angle)
        sin = np.sin(angle)
        return torch.tensor([sin, cos], dtype=torch.float32)


class RotaryPositionalEmbedding(nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, rope: RoPE, device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.rope = rope

    def forward(self, in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
                token_positions: Int[Tensor, " ... sequence_length"]) -> torch.Tensor:
        R = self.rope.get_buffer("R")[token_positions]
        q = rearrange(in_query_or_key, " ... sequence_length (k pair) ->  ... sequence_length k pair", pair=2)
        x_2k = q[..., 0] * R[..., 1] - q[..., 1] * R[..., 0]
        x_2kp1 = q[..., 0] * R[..., 0] + q[..., 1] * R[..., 1]
        new_q = torch.stack([x_2k, x_2kp1], dim=-1)
        new_q = rearrange(new_q, " ... sequence_length k pair ->  ... sequence_length (k pair)")
        return new_q
