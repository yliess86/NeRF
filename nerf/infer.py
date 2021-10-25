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
    H: int,
    W: int,
    batch_size: int = 32,
    verbose: bool = True,
) -> Tensor:
    """Neural radiance field inference (render frame)

    Arguments:
        nerf (NeRF): neural radiance field model
        raymarcher (BVR): bounded volume raymarcher
        ro (Tensor): rays origin (B, 3)
        rd (Tensor): rays direction (B, 3)
        H (int): frame height
        W (int): frame width
        batch_size (int): batch size
        verbose (bool): print tqdm

    Returns:
        depth_map (Tensor): rendered depth map [0, 255] (H, W, 3)
        rgb_map (Tensor): rendered rgb map [0, 255] (H, W, 3)
    """
    d = next(nerf.parameters()).device
    n = ro.size(0)
    
    ro, rd = ro.to(d), rd.to(d)

    depth_map = torch.zeros((H * W, ), dtype=torch.float32)
    rgb_map = torch.zeros((H * W, 3), dtype=torch.float32)

    batches = range(0, n, batch_size)
    for s in tqdm(batches, desc="[NeRF] Rendering", disable=not verbose):
        e = min(s + batch_size, n)
        *_, D, C = raymarcher.render_volume(nerf, ro[s:e], rd[s:e], train=False)
        
        depth_map[s:e] = D.cpu()
        rgb_map[s:e] = C.cpu()

    depth_map = depth_map.view(H, W, 1)
    depth_map = depth_map.repeat((1, 1, 3))
    depth_map = depth_map - depth_map.min()
    depth_map = depth_map / (depth_map.max() + 1e-10)
    depth_map = depth_map.clip(0, 1) * 255

    rgb_map = rgb_map.view(H, W, 3)
    rgb_map = rgb_map.clip(0, 1) * 255
    
    return depth_map, rgb_map


NeRF.infer = infer