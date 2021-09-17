import torch
import torch.jit as jit

from torch import device, Tensor
from typing import Tuple


INF = 1e10


@jit.script
def uniform_bounded_z_values(
    tn: float,
    tf: float,
    batch_size: int,
    samples: int,
    d: device,
) -> Tensor:
    """Generate uniform bounded z-values

    Arguments:
        tn (float): near plane
        tf (float): far plane
        batch_size (int): batch size B
        samples (int): number of samples N
        d (device): torch device

    Returns:
        t (Tensor): z-values from near to far (B, N)
    """
    t = torch.linspace(0, 1, samples, device=d)
    t = tn * (1 - t) + tf * t
    return t.expand(batch_size, samples)


@jit.script
def segment_lengths(t: Tensor, rd: Tensor) -> Tensor:
    """Compute rays segment length

    Arguments:
        t (Tensor): z-values from near to far (B, N)
        rd (Tensor): rays direction (B, 3)

    Returns:
        delta (Tensor): rays segment length (B, N)
    """
    delta = t[:, 1:] - t[:, :-1]
    delti = INF * torch.ones_like(delta)
    delta = torch.cat((delta, delti), dim=-1)
    delta = delta * torch.norm(rd[:, None, :], dim=-1)


@jit.script
def uniform_bounded_rays(
    ro: Tensor,
    rd: Tensor,
    tn: float,
    tf: float,
    samples: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Sample along rays uniformly in bounded volume

    Arguments:
        ro (Tensor): rays origin (B, 3)
        rd (Tensor): rays direction (B, 3)
        tn (float): near plane
        tf (float): far plane
        samples (int): number of samples along the ray

    Returns:
        rx (Tensor): rays position queries (B, N, 3)
        rd (Tensor): rays direction (B, N, 3)
        t (Tensor): z-values from near to far (B, N)
        delta (Tensor): rays segment lengths (B, N)
    """
    B, N = ro.size(0), samples

    t = uniform_bounded_z_values(tn, tf, B, N, ro.device)
    delta = segment_lengths(t, rd)

    rx = ro[:, None, :] + rd[:, None, :] * t[:, :, None]
    rd = torch.repeat_interleave(rd, repeats=N, dim=0)

    return rx, rd, t, delta