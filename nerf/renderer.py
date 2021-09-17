import torch
import torch.jit as jit

from torch import Tensor
from typing import Tuple


EPS = 1e-10


@jit.script
def exclusive_cumprod(x: Tensor) -> Tensor:
    """Tensorflow tf.math.cumprod(..., exclusive=True) equivalent

    tf.math.cumprod([a, b, c], exclusive=True) = [1, a, a * b]
    default implementation for last dimension

    Arguments:
        x (Tensor): input tensor (B, N)

    Returns:
        cp (Tensor): output tensor (B, N)
    """
    cp = torch.cumprod(x, dim=-1)
    cp = torch.roll(cp, 1, dims=-1)
    cp[..., 0] = 1.
    return cp


@jit.script
def render_volume(
    sigma: Tensor,
    rgb: Tensor,
    delta: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Render volume given radiance field data
    
    Arguments:
        sigma (Tensor): density at volume query (B, N)
        rgb (Tensor): color at volume query (B, N, 3)
        delta (Tensor): segments lengths (B, N)

    Returns:
        w (Tensor): absorbtion weights for each ray (B, N)
        C (Tensor): accumulated render color for each ray (B, 3)
    """
    alpha = 1 - torch.exp(-sigma * delta)
    w = alpha * exclusive_cumprod(1 - alpha + EPS, dim=-2)
    return w, torch.sum(w[..., None] * rgb, dim=-2)