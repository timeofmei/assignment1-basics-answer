import torch
import torch.nn as nn
import numpy as np
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor
from rope import RotaryPositionalEmbedding
from scaled_dot_product_attention import scaled_dot_product_attention


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, rope_layer: RotaryPositionalEmbedding | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.rope_layer = rope_layer
        self.device = device
        self.dtype = dtype

        W_Q = torch.zeros((d_model, d_model), dtype=torch.float, device=device)
        W_K = torch.zeros((d_model, d_model), dtype=torch.float, device=device)
        W_V = torch.zeros((d_model, d_model), dtype=torch.float, device=device)
        W_O = torch.zeros((d_model, d_model), dtype=torch.float, device=device)
        self.W_Q = nn.Parameter(W_Q, requires_grad=True)
        self.W_K = nn.Parameter(W_K, requires_grad=True)
        self.W_V = nn.Parameter(W_V, requires_grad=True)
        self.W_O = nn.Parameter(W_O, requires_grad=True)

    def load_weights(self, weights: dict[str, Tensor]):
        self.load_state_dict(weights, strict=False)

    def forward(self, x: Float[Tensor, "... sequence_length d_model"]) -> Float[Tensor, " ... sequence_length d_model"]:
        seq_length = x.shape[-2]
        W_Q_multi = rearrange(self.W_Q, "(h d_k) d_model -> h d_k d_model", h=self.num_heads)
        W_K_multi = rearrange(self.W_K, "(h d_k) d_model -> h d_k d_model", h=self.num_heads)
        W_V_multi = rearrange(self.W_V, "(h d_v) d_model -> h d_v d_model", h=self.num_heads)
        W_O_multi = rearrange(self.W_O, "d_model (h d_v) -> d_model h d_v", h=self.num_heads)
        W_QKV_stack = torch.stack([W_Q_multi, W_K_multi, W_V_multi], dim=0)
        QKV_stack = einsum(W_QKV_stack, x, "triple h d_k d_model, ... sequence_length d_model -> ... triple h sequence_length d_k")
        Q_multi, K_multi, V_multi = rearrange(QKV_stack, "... triple h sequence_length d_k -> triple ... h sequence_length d_k")
        seq_indexes = torch.arange(0, seq_length)
        if self.rope_layer is not None:
            Q_multi = self.rope_layer(Q_multi, seq_indexes)
            K_multi = self.rope_layer(K_multi, seq_indexes)
        single_mask = ~torch.ones((seq_length, seq_length)).triu(1).to(torch.bool)
        attn_val = scaled_dot_product_attention(Q_multi, K_multi, V_multi, single_mask)
        result = einsum(W_O_multi, attn_val, "d_model h d_v, ... h sequence_length d_v ->  ... sequence_length d_model")
        return result
