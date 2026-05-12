import torch
import numpy as np
from einops import einsum
from jaxtyping import Float, Bool
from torch import Tensor
from softmax import softmax


def scaled_dot_product_attention(Q: Float[Tensor, " ... queries d_k"], K: Float[Tensor, " ... keys d_k"], V: Float[Tensor, " ... keys d_v"], mask: Bool[Tensor, " ... queries keys"] | None = None) -> Float[Tensor, " ... queries d_v"]:
    sqrt_d_k = np.sqrt(Q.shape[-1])
    pre_softmax = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / sqrt_d_k
    if mask is not None:
        mask_val = torch.zeros(mask.shape).masked_fill(~mask, -torch.inf)
        pre_softmax += mask_val
    attention_softmax_val = softmax(pre_softmax, -1)
    attention_val = einsum(attention_softmax_val, V, "... queries keys, ... keys d_v -> ... queries d_v")
    return attention_val
