import torch

from nerf.core.model import NeRF
from nerf.core.renderer import BoundedVolumeRaymarcher as BVR
from nerf.utils.pbar import tqdm
from torch import Tensor


@torch.inference_mode()
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
        pred (Tensor): rendered frame [0, 255] (H, W, 3)
    """
    d = next(nerf.parameters()).device
    n = ro.size(0)
    
    ro, rd = ro.to(d), rd.to(d)
    pred = torch.zeros((H * W, 3), dtype=torch.float32)

    batches = range(0, n, batch_size)
    for s in tqdm(batches, desc="[NeRF] Rendering", disable=not verbose):
        e = min(s + batch_size, n)
        *_, C = raymarcher.render_volume(nerf, ro[s:e], rd[s:e], train=False)
        pred[s:e] = C.cpu()

    return pred.view(H, W, 3).clip(0, 1) * 255


NeRF.infer = infer