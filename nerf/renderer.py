import torch
import torch.jit as jit

from nerf.model import NeRF
from nerf.ray import uniform_bounded_rays as ubrays
from torch import Tensor
from typing import Tuple


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
    EPS = 1e-10
    
    alpha = 1 - torch.exp(-sigma * delta)
    w = alpha * exclusive_cumprod(1 - alpha + EPS)
    return w, torch.sum(w[:, :, None] * rgb, dim=-2)


class BoundedVolumeRaymarcher:
    """Bounded volume raymarcher

    Arguments:
        tn (float): near plane
        tf (float): far plane
        samples (int): number of samples along the ray (default: 64)
    """

    def __init__(self, tn: float, tf: float, samples: int = 64) -> None:
        self.tn = tn
        self.tf = tf
        self.samples = samples

    def render_volume(
        self,
        nerf: NeRF,
        ro: Tensor,
        rd: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Render implicit volume given ray infos

        Arguments:
            nerf (NeRF): query Neural Radiance Field model
            rx (Tensor): ray query position (B, 3)
            rd (Tensor): ray query direction (B, 3)

        Returns:
            C (Tensor): accumulated render color for each ray (B, 3)
        """
        B, N = ro.size(0), self.samples

        rx, rd, _, delta = ubrays(ro, rd, self.tn, self.tf, N)
        rx = rx.view(B * N, 3)
        rd = rx.view(B * N, 3)

        sigma, rgb = nerf(rx, rd)
        sigma = sigma.view(B, N)
        rgb = rgb.view(B, N, 3)
        
        _, C = render_volume(sigma, rgb, delta)
        
        return C