import numpy as np
import numpy.typing as npt
import torch


def get_batch(
        dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    input_seq = []
    pair_seq = []
    n = len(dataset)
    for _ in range(batch_size):
        i = np.random.choice(n - context_length)
        input_seq.append(torch.tensor(dataset[i:i+context_length], device=device))
        pair_seq.append(torch.tensor(dataset[i+1:i+1+context_length], device=device))
    return torch.stack(input_seq), torch.stack(pair_seq)
