import torch

from nerf.core.model import NeRF
from nerf.core.renderer import BoundedVolumeRaymarcher as BVR
from torch import Tensor
from tqdm.auto import tqdm


def infer(
    nerf: NeRF,
    raymarcher: BVR,
    ro: Tensor,
    rd: Tensor,
    W: int,
    H: int,
    batch_size: int = 32,
    verbose: bool = True,
) -> Tensor:
    """Neural radiance field inference (render frame)

    Arguments:
        nerf (NeRF): neural radiance field model
        raymarcher (BVR): bounded volume raymarcher
        ro (Tensor): rays origin (B, 3)
        rd (Tensor): rays direction (B, 3)
        W (int): frame width
        H (int): frame height
        batch_size (int): batch size
        verbose (bool): print tqdm

    Returns:
        pred (Tensor): rendered frame [0, 255] (W, H, 3)
    """
    nerf = nerf.eval()

    d = next(nerf.parameters()).device
    n = len(ro)

    with torch.inference_mode():
        pred = []
        batches = range(0, n, batch_size)

        pbar = tqdm(batches, desc="[NeRF] Rendering", disable=(not verbose))
        for s in pbar:
            e = min(s + batch_size, n)
            rays = ro[s:e].to(d), rd[s:e].to(d)
            C = raymarcher.render_volume(nerf, *rays)
            pred.append(C.cpu())

        pred = torch.cat(pred, dim=0).view(W, H, 3)
        pred = pred.clip(0, 1) * 255

    return pred


NeRF.infer = infer