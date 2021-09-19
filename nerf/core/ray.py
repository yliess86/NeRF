import torch
import torch.jit as jit

from torch import device, Tensor
from typing import Tuple


@jit.script
def pinhole_ray_directions(W: int, H: int, focal: float) -> Tensor:
    """Generate pinhole camera ray directions from origin to pixels

    Arguments:
        W (int): frame width
        H (int): frame height
        focal (float): camera focal length

    Returns:
        rd (Tensor): ray directions (W, H, 3)
    """
    Ws = torch.linspace(0, W - 1, W)
    Hs = torch.linspace(0, H - 1, H)
    i, j = torch.meshgrid(Ws, Hs)
    rdx =  (i.t() - .5 * W) / focal
    rdy = -(j.t() - .5 * H) / focal
    rdz = -torch.ones_like(i.t())
    return torch.stack((rdx, rdy, rdz), dim=-1)


@jit.script
def phinhole_ray_projection(
    prd: Tensor, c2w: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Project pinhole camera rays from camera to world

    Arguments:
        prd (Tensor): ray directions in camera coords (W, H, 3)
        c2w (Tensor): camera to world projection matrix (4, 4)

    Returns:
        ro (Tensor): ray origin in world coords (W, H, 3)
        rd (Tensor): ray directions in world coords (W, H, 3)
    """
    rd = prd @ c2w[:3, :3].T
    rd = rd / torch.norm(rd, dim=-1, keepdim=True)
    ro = c2w[:3, 3].expand(rd.size())
    return ro, rd


@jit.script
def uniform_bounded_z_values(
    tn: float,
    tf: float,
    batch_size: int,
    samples: int,
    d: device,
    perturb: bool,
) -> Tensor:
    """Generate uniform bounded z-values

    Arguments:
        tn (float): near plane
        tf (float): far plane
        batch_size (int): batch size B
        samples (int): number of samples N
        d (device): torch device
        perturb (bool): peturb ray query segment

    Returns:
        t (Tensor): z-values from near to far (B, N)
    """
    t = torch.linspace(tn, tf, samples, device=d)
    if perturb: t += torch.randn_like(t) * (tf - tn) / samples
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
    B, INF = t.size(0), 1e10

    delta = t[:, 1:] - t[:, :-1]
    delti = INF * torch.ones((B, 1), device=rd.device)
    delta = torch.cat((delta, delti), dim=-1)
    delta = delta * torch.norm(rd[:, None, :], dim=-1)
    return delta


@jit.script
def uniform_bounded_rays(
    ro: Tensor,
    rd: Tensor,
    tn: float,
    tf: float,
    samples: int,
    perturb: bool,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Sample along rays uniformly in bounded volume

    Arguments:
        ro (Tensor): rays origin (B, 3)
        rd (Tensor): rays direction (B, 3)
        tn (float): near plane
        tf (float): far plane
        samples (int): number of samples along the ray
        perturb (bool): peturb ray query segment

    Returns:
        rx (Tensor): rays position queries (B, N, 3)
        rd (Tensor): rays direction (B, N, 3)
        t (Tensor): z-values from near to far (B, N)
        delta (Tensor): rays segment lengths (B, N)
    """
    B, N = ro.size(0), samples

    t = uniform_bounded_z_values(tn, tf, B, N, ro.device, perturb)
    delta = segment_lengths(t, rd)

    rx = ro[:, None, :] + rd[:, None, :] * t[:, :, None]
    rd = torch.repeat_interleave(rd, repeats=N, dim=0)

    return rx, rd, t, delta