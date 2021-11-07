import torch
import torch.jit as jit

from nerf.core.model import NeRF
from nerf.core.ray import pdf_rays as prays, uniform_bounded_rays as ubrays
from torch import Tensor
from typing import Optional, Tuple, Union


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
    weights_only: bool,
) -> Tuple[Tensor, Union[Tensor, None]]:
    """Raymarch into volume given radiance field data
    
    Arguments:
        sigma (Tensor): density at volume query (B, N)
        rgb (Tensor): color at volume query (B, N, 3)
        delta (Tensor): segments lengths (B, N)
        weights_only (bool): compute weights only (rgb_map will be set to None)

    Returns:
        weights (Tensor): absorbtion weights for each ray (B, N)
        rgb_map (Tensor): accumulated render color for each ray (B, 3)
    """
    EPS = 1e-10
    
    alpha = 1. - torch.exp(-sigma * delta)
    trans = exclusive_cumprod(1. - alpha + EPS)
    weights = alpha * trans

    if weights_only:
        return weights, None
    
    return weights, torch.sum(weights[:, :, None] * rgb, dim=-2)


def render_volume_coarse(
    nerf: NeRF,
    ro: Tensor,
    rd: Tensor,
    tn: float, 
    tf: float, 
    samples: int,
    perturb: bool,
    weights_only: bool,
) -> Tuple[Tensor, Tensor, Union[Tensor, None]]:
    """Render implicit coarse volume given ray infos

    Arguments:
        nerf (NeRF): query Neural Radiance Field model
        ro (Tensor): ray query origin (B, 3)
        rd (Tensor): ray query direction (B, 3)
        tn (float): near plane
        tf (float): far plane
        samples (int): number of coarse samples along the ray
        perturb (bool): peturb ray query segment
        weights_only (bool): compute weights only (rgb_map will be set to None)

    Returns:
        t (Tensor): z-values from near to far (B, N)
        weights (Tensor): absorbtion weights for each ray (B, N)
        rgb_map (Tensor): accumulated render color for each ray (B, 3)
    """
    B, Nc = ro.size(0), samples

    rx, rds, t, delta = ubrays(ro, rd, tn, tf, Nc, perturb)

    sigma, rgb = nerf(rx, rds)
    sigma = sigma.view(B, Nc)
    rgb = rgb.view(B, Nc, 3)

    weights, rgb_map = raymarch_volume(sigma, rgb, delta, weights_only)

    return t, weights, rgb_map


def render_volume_fine(
    coarse: NeRF,
    fine: NeRF,
    ro: Tensor,
    rd: Tensor,
    tn: float, 
    tf: float, 
    samples_c: int,
    samples_f: int,
    perturb: bool,
    train: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Render implicit refined volume given ray infos

    Arguments:
        coarse (NeRF): coarse query Neural Radiance Field model
        fine (NeRF): fine query Neural Radiance Field model
        ro (Tensor): ray query origin (B, 3)
        rd (Tensor): ray query direction (B, 3)
        tn (float): near plane
        tf (float): far plane
        samples_c (int): number of coarse samples along the ray
        samples_f (int): number of fine samples along the ray
        perturb (bool): peturb ray query segment
        train (bool): train or eval mode

    Returns:
        t (Tensor): z-values from near to far (B, N)
        weights (Tensor): absorbtion weights for each ray (B, N)
        rgb_map (Tensor): accumulated render color for each ray (B, 3)
    """
    B, Nc, Nf = ro.size(0), samples_c, samples_f

    if train: coarse.requires_grad(False)
    t, weights, _ = render_volume_coarse(coarse, ro, rd, tn ,tf, Nc, perturb, weights_only=True)
    rx, rds, t, delta = prays(ro, rd, t, weights, Nf, perturb)

    if train: fine.requires_grad(True)
    sigma, rgb = fine(rx, rds)
    sigma = sigma.view(B, Nc + Nf)
    rgb = rgb.view(B, Nc + Nf, 3)

    weights, rgb_map = raymarch_volume(sigma, rgb, delta, weights_only=False)

    return t, weights, rgb_map


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
        coarse: NeRF,
        fine: NeRF,
        ro: Tensor,
        rd: Tensor,
        perturb: Optional[bool] = False,
        train: Optional[bool] = False,
    ) -> Tuple[Tensor, Tensor]:
        """Render implicit volume given ray infos

        Arguments:
            coarse (NeRF): coarse query Neural Radiance Field model
            fine (NeRF): fine query Neural Radiance Field model
            ro (Tensor): ray query origin (B, 3)
            rd (Tensor): ray query direction (B, 3)
            perturb (Optional[bool]): peturb ray query segment (default: False)
            train (Optional[bool]): train or eval mode (default: False)

        Returns:
            t (Tensor): z-values from near to far (B, N)
            weights (Tensor): absorbtion weights for each ray (B, N)
            depth_map (Tensor): depth map for each ray (B, )
            rgb_map (Tensor): accumulated render color for each ray (B, 3)
        """
        Nc, Nf = self.samples_c, self.samples_f
        bounds = self.tn, self.tf
        
        if Nf > 0: t, weights, rgb_map = render_volume_fine(coarse, fine, ro, rd, *bounds, Nc, Nf, perturb, train)
        else: t, weights, rgb_map = render_volume_coarse(coarse, ro, rd, *bounds, Nc, perturb, False)
        depth_map = torch.sum(weights * t, dim=-1)

        return t, weights, depth_map, rgb_map