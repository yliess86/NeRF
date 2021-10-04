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
    if perturb: t += torch.rand_like(t) * (tf - tn) / samples
    return t.expand(batch_size, samples).contiguous()


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
    rd = torch.repeat_interleave(rd, repeats=N, dim=0).view(B, N, 3)

    return rx, rd, t, delta


@jit.script
def pdf_z_values(
    bins: Tensor,
    weights: Tensor,
    samples: int,
    d: device,
    perturb: bool,
) -> Tensor:
    """Generate z-values from pdf

    Arguments:
        bins (Tensor): z-value bins (B, N - 2)
        weights (Tensor): bin weights gathered from first pass (B, N - 1)
        samples (int): number of samples N
        d (device): torch device
        perturb (bool): peturb ray query segment

    Returns:
        t (Tensor): z-values sampled from pdf (B, N)
    """
    EPS = 1e-10
    B, N = weights.size()

    weights = weights + EPS
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat((torch.zeros_like(cdf[:, :1]), cdf), dim=-1)
    cdf = cdf.contiguous()

    if perturb:
        u = torch.rand((B, samples), device=d)
        u = u.contiguous()
    else:
        u = torch.linspace(0, 1, samples, device=d)
        u = u.expand(B, samples)
        u = u.contiguous()

    idxs = torch.searchsorted(cdf, u, right=True)
    idxs_below = torch.clamp_min(idxs - 1, 0)
    idxs_above = torch.clamp_max(idxs, N)
    idxs = torch.stack((idxs_below, idxs_above), dim=-1).view(B, 2 * samples)

    cdf = torch.gather(cdf, dim=1, index=idxs).view(B, samples, 2)
    bins = torch.gather(bins, dim=1, index=idxs).view(B, samples, 2)
    
    den = cdf[:, :, 1] - cdf[:, :, 0]
    den[den < EPS] = 1.

    t = (u - cdf[:, :, 0]) / den
    t = bins[:, :, 0] + t * (bins[:, :, 1] - bins[:, :, 0])
    
    return t


@jit.script
def pdf_rays(
    ro: Tensor,
    rd: Tensor,
    t: Tensor,
    weights: Tensor,
    samples: int,
    perturb: bool,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Sample pdf along rays given computed weights

    Arguments:
        ro (Tensor): rays origin (B, 3)
        rd (Tensor): rays direction (B, 3)
        t (Tensor): coarse z-value (B, N)
        weights (Tensor): weights gathered from first pass (B, N)
        samples (int): number of samples along the ray
        perturb (bool): peturb ray query segment

    Returns:
        rx (Tensor): rays position queries (B, Nc + Nf, 3)
        rd (Tensor): rays direction (B, Nc + Nf, 3)
        t (Tensor): z-values from near to far (B, Nc + Nf)
        delta (Tensor): rays segment lengths (B, Nc + Nf)
    """
    B, Nc = weights.size()
    Nf = samples
    N = Nc + Nf

    b = .5 * (t[:, :-1] - t[:, 1:])
    w = weights[:, 1:-1]

    z = pdf_z_values(b, w, Nf, ro.device, perturb)
    t, _ = torch.sort(torch.cat((t, z), dim=-1), dim=-1)
    delta = segment_lengths(t, rd)

    rx = ro[:, None, :] + rd[:, None, :] * t[:, :, None]
    rd = torch.repeat_interleave(rd, repeats=N, dim=0).view(B, N, 3)

    return rx, rd, t, delta