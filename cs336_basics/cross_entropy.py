import torch
from torch import Tensor
from jaxtyping import Float, Int


def cross_entropy(logit: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    logit -= logit.max(dim=-1, keepdim=True).values
    target_logit = torch.gather(logit, -1, targets.unsqueeze(-1))
    logsumexp = logit.exp().sum(-1, keepdim=True).log()
    result = logsumexp - target_logit
    return result.mean()
