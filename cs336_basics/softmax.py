import torch
from jaxtyping import Float
from torch import Tensor


def softmax(x: Float[Tensor, " ..."], i: int):
    max_val = x.max(i, keepdim=True).values
    stable_x = x - max_val
    stable_exp_x = stable_x.exp()
    softmax_x = stable_exp_x / stable_exp_x.sum(i, keepdim=True)
    return softmax_x
