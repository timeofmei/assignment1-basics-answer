import torch
import os
from typing import IO, BinaryIO

def save_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]):
    save = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "iteration": iteration}
    torch.save(save, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    save = torch.load(src)
    model.load_state_dict(save["model_state"])
    optimizer.load_state_dict(save["optimizer_state"])
    return save["iteration"]