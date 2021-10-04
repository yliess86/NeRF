import torch
import torch.jit as jit

from nerf.core.model import NeRF
from nerf.core.ray import pdf_rays as prays, uniform_bounded_rays as ubrays
from torch import Tensor
from typing import Optional, Tuple


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
def raymarch_volume(
    sigma: Tensor,
    rgb: Tensor,
    delta: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Raymarch into volume given radiance field data
    
    Arguments:
        sigma (Tensor): density at volume query (B, N)
        rgb (Tensor): color at volume query (B, N, 3)
        delta (Tensor): segments lengths (B, N)

    Returns:
        w (Tensor): absorbtion weights for each ray (B, N)
        C (Tensor): accumulated render color for each ray (B, 3)
    """
    EPS = 1e-10
    
    alpha = 1 - torch.exp(-sigma * delta)
    w = alpha * exclusive_cumprod(1. - alpha + EPS)
    return w, torch.sum(w[:, :, None] * rgb, dim=-2)


def render_volume_coarse(
    nerf: NeRF,
    ro: Tensor,
    rd: Tensor,
    tn: float, 
    tf: float, 
    samples: int,
    perturb: bool,
) -> Tensor:
    """Render implicit coarse volume given ray infos

    Arguments:
        nerf (NeRF): query Neural Radiance Field model
        rx (Tensor): ray query position (B, 3)
        rd (Tensor): ray query direction (B, 3)
        tn (float): near plane
        tf (float): far plane
        samples (int): number of coarse samples along the ray
        perturb (bool): peturb ray query segment

    Returns:
        C (Tensor): accumulated render color for each ray (B, 3)
    """
    B, Nc = ro.size(0), samples

    rx, rds, _, delta = ubrays(ro, rd, tn, tf, Nc, perturb)

    sigma, rgb = nerf(rx, rds)
    sigma = sigma.view(B, Nc)
    rgb = rgb.view(B, Nc, 3)

    _, C = raymarch_volume(sigma, rgb, delta)
    
    return C


def render_volume_fine(
    nerf: NeRF,
    ro: Tensor,
    rd: Tensor,
    tn: float, 
    tf: float, 
    samples_c: int,
    samples_f: int,
    perturb: bool,
    train: bool,
) -> Tensor:
    """Render implicit refined volume given ray infos

    Arguments:
        nerf (NeRF): query Neural Radiance Field model
        rx (Tensor): ray query position (B, 3)
        rd (Tensor): ray query direction (B, 3)
        tn (float): near plane
        tf (float): far plane
        samples_c (int): number of coarse samples along the ray
        samples_f (int): number of fine samples along the ray
        perturb (bool): peturb ray query segment
        train (bool): train or eval mode

    Returns:
        C (Tensor): accumulated render color for each ray (B, 3)
    """
    B, Nc, Nf, N = ro.size(0), samples_c, samples_f, samples_c + samples_f

    rx, rds, t, delta = ubrays(ro, rd, tn, tf, Nc, perturb)

    if train: nerf.requires_grad(False)

    rx = rx.view(B * Nc, 3)
    rds = rds.view(B * Nc, 3)

    sigma, rgb = nerf(rx, rds)
    sigma = sigma.view(B, Nc)
    rgb = rgb.view(B, Nc, 3)

    w, _ = raymarch_volume(sigma, rgb, delta)

    rx, rds, _, delta = prays(ro, rd, t, w, Nf, perturb)
    rx = rx.view(B * N, 3)
    rds = rds.view(B * N, 3)

    if train: nerf.requires_grad(True)

    sigma, rgb = nerf(rx, rds)
    sigma = sigma.view(B, N)
    rgb = rgb.view(B, N, 3)

    _, C = raymarch_volume(sigma, rgb, delta)
    
    return C


class BoundedVolumeRaymarcher:
    """Bounded volume raymarcher

    Arguments:
        tn (float): near plane
        tf (float): far plane
        samples_c (int): number of coarse samples along the ray (default: 64)
        samples_f (int): number of fine samples along the ray (default: 64)
    """

    def __init__(
        self,
        tn: float,
        tf: float,
        samples_c: int = 64,
        samples_f: int = 64,
    ) -> None:
        assert tf > tn, "[NeRF] `tf` should always be > `tn`"
        assert samples_c > 0, "[NeRF] `samples_c` should always be > 0"

        self.tn = tn
        self.tf = tf
        self.samples_c = samples_c
        self.samples_f = samples_f
        self.samples = self.samples_c + self.samples_f

    def render_volume(
        self,
        nerf: NeRF,
        ro: Tensor,
        rd: Tensor,
        perturb: Optional[bool] = False,
        train: Optional[bool] = False,
    ) -> Tensor:
        """Render implicit volume given ray infos

        Arguments:
            nerf (NeRF): query Neural Radiance Field model
            rx (Tensor): ray query position (B, 3)
            rd (Tensor): ray query direction (B, 3)
            perturb (Optional[bool]): peturb ray query segment (default: False)
            train (Optional[bool]): train or eval mode (default: False)

        Returns:
            C (Tensor): accumulated render color for each ray (B, 3)
        """
        Nc, Nf = self.samples_c, self.samples_f
        
        if Nf > 0:
            return render_volume_fine(nerf, ro, rd, self.tn, self.tf, Nc, Nf, perturb, train)
        return render_volume_coarse(nerf, ro, rd, self.tn, self.tf, Nc, perturb)