import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor
from .linear import Linear
from .rope import RotaryPositionalEmbedding
from .scaled_dot_product_attention import scaled_dot_product_attention


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, rope_layer: RotaryPositionalEmbedding | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.rope_layer = rope_layer
        self.device = device
        self.dtype = dtype

        self.q_proj = Linear(d_model, d_model, dtype=torch.float, device=device)
        self.k_proj = Linear(d_model, d_model, dtype=torch.float, device=device)
        self.v_proj = Linear(d_model, d_model, dtype=torch.float, device=device)
        self.output_proj = Linear(d_model, d_model, dtype=torch.float, device=device)

    def load_weight(self, q_proj_weight: Float[Tensor, " d_model d_model"], k_proj_weight: Float[Tensor, " d_model d_model"], v_proj_weight: Float[Tensor, " d_model d_model"], o_proj_weight: Float[Tensor, " d_model d_model"]):
        weight = {"q_proj.weight": q_proj_weight, "k_proj.weight": k_proj_weight, "v_proj.weight": v_proj_weight, "output_proj.weight": o_proj_weight}
        self.load_state_dict(weight, strict=False)

    def forward(self, x: Float[Tensor, "... sequence_length d_model"], token_positions: Int[Tensor, " ... sequence_length"] | None = None) -> Float[Tensor, " ... sequence_length d_model"]:
        seq_length = x.shape[-2]
        Q_multi = rearrange(self.q_proj(x), "... sequence_length (h d_k) -> ... h sequence_length d_k", h=self.num_heads)
        K_multi = rearrange(self.k_proj(x), "... sequence_length (h d_k) -> ... h sequence_length d_k", h=self.num_heads)
        V_multi = rearrange(self.v_proj(x), "... sequence_length (h d_v) -> ... h sequence_length d_v", h=self.num_heads)
        if token_positions is None:
            token_positions = torch.arange(0, seq_length, device=self.device)
        if self.rope_layer is not None:
            Q_multi = self.rope_layer(Q_multi, token_positions)
            K_multi = self.rope_layer(K_multi, token_positions)
        single_mask = ~torch.ones((seq_length, seq_length), device=self.device).triu(1).to(torch.bool)
        attn_val = scaled_dot_product_attention(Q_multi, K_multi, V_multi, single_mask)
        attn_val = rearrange(attn_val, "... h sequence_length d_v -> ... sequence_length (h d_v)")
        result = self.output_proj(attn_val)
        return result
