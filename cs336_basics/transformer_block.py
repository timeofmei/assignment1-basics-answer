import torch
import torch.nn as nn
import numpy as np
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor
from .rope import RoPE, RotaryPositionalEmbedding
from .scaled_dot_product_attention import scaled_dot_product_attention
from .multihead_self_attention import MultiHeadSelfAttention
from .positionwise_feedforward import SwiGLU
from .rmsnorm import RMSNorm


class TransformerBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        self.dtype = dtype

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.rope = RoPE(theta, self.d_k, max_seq_len, device=device)
        self.rope_layer = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, self.rope, device=device)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, self.rope_layer, device=device, dtype=dtype)

        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def load_weight(self, weight: dict[str, Tensor]):
        self.load_state_dict(weight, strict=False)

    def forward(self, x: Float[Tensor, "... sequence_length d_model"]) -> Float[Tensor, " ... sequence_length d_model"]:
        residual_1 = x
        x = self.ln1(x)
        x = self.attn(x)
        x += residual_1
        residual_2 = x
        x = self.ln2(x)
        x = self.ffn(x)
        x += residual_2
        return x
