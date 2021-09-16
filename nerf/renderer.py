import torch
import torch.jit as jit

from torch import Tensor
from typing import Tuple


EPS = 1e-10


@jit.script
def exclusive_cumprod(x: Tensor, dim: int = -1) -> Tensor:
    cp = torch.cumprod(x, dim=dim)
    cp = torch.roll(cp, 1, dims=dim)
    cp[..., 0] = 1.
    return cp


@jit.script
def render_volume(
    sigma: Tensor,
    rgb: Tensor,
    delta: Tensor,
) -> Tuple[Tensor, Tensor]:
    alpha = 1 - torch.exp(-sigma * delta)
    w = alpha * exclusive_cumprod(1 - alpha + EPS, dim=-1)
    return w, torch.sum(w[..., None] * rgb, dim=-2)