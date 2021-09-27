import gc
import torch

from copy import deepcopy
from nerf.core.model import NeRF
from nerf.core.renderer import BoundedVolumeRaymarcher as BVR
from nerf.utils.pbar import tqdm
from torch import device
from torch.cuda.amp import autocast, GradScaler
from torch.nn import Module
from torch.optim import Optimizer, SGD
from torch.utils.data import DataLoader
from typing import Optional, Tuple


def build_meta(
    nerf: NeRF,
    optim: Optimizer,
    scaler: GradScaler,
) -> Tuple[NeRF, Optimizer, GradScaler]:
    """Build Meta Learning Objects

    Arguments:
        nerf (NeRF): neural radiance field model to be trained
        optim (Optimizer): optimization strategy
        scaler (GradScaler): grad scaler for half precision (fp16)

    Returns:
        meta_nerf (NeRF): meta neural radiance field model to be trained
        meta_optim (Optimizer): meta optimizer strategy
        meta_scaler (GradScaler): meta grad scaler for half precision (fp16)
    """
    optim_lr = optim.param_groups[0]["lr"]

    meta_nerf = deepcopy(nerf)
    meta_optim = SGD(meta_nerf.parameters(), lr=optim_lr * 1e3, momentum=0.9)
    meta_scaler = deepcopy(scaler)

    return meta_nerf, meta_optim, meta_scaler


class MetaStateHolder:
    """Meta State Holder for Weight Reset/Update

    Arguments:
        nerf (NeRF): neural radiance field model to be trained
        meta_nerf (NeRF): meta neural radiance field model to be trained
        meta_optim (Optimizer): meta optimization strategy
        meta_scaler (GradScaler): meta grad scaler for half precision (fp16)
    """

    def __init__(self,
        nerf: NeRF,
        meta_nerf: NeRF,
        meta_optim: Optimizer,
        meta_scaler: GradScaler,
    ) -> None:
        self.nerf = nerf
        self.meta_nerf = meta_nerf
        self.meta_optim = meta_optim
        self.meta_scaler = meta_scaler

        self.meta_optim_is = self.meta_optim.state_dict()
        self.meta_scaler_is = self.meta_scaler.state_dict()

    @torch.no_grad()
    def update(self) -> None:
        """Reset/Update States"""
        self.meta_nerf.load_state_dict(self.nerf.state_dict())
        self.meta_optim.load_state_dict(self.meta_optim_is)
        self.meta_scaler.load_state_dict(self.meta_scaler_is)


def meta_initialization(
    nerf: NeRF,
    raymarcher: BVR,
    optim: Optimizer,
    criterion: Module,
    scaler: GradScaler,
    loader: DataLoader,
    d: device,
    perturb: Optional[bool] = False,
    meta_steps: Optional[int] = 16,
    verbose: Optional[bool] = True,
) -> Tuple[float, float]:
    """Meta Initialization

    Arguments:
        nerf (NeRF): neural radiance field model to be trained
        raymarcher (BVR): bounded volume raymarching renderer
        optim (Optimizer): optimization strategy
        criterion (Module): objective function
        scaler (GradScaler): grad scaler for half precision (fp16)
        loader (DataLoader): batch data loader
        d (device): torch device to send the batch on
        perturb (Optional[bool]): peturb ray query segment (default: False)
        meta_steps (Optional[int]): number of meta steps to perform before updating (default: 16)
        verbose (Optional[bool]): print tqdm (default: True)

    Returns:
        total_loss (float): arveraged cumulated total loss
        total_psnr (float): arveraged cumulated total psnr
    """
    nerf = nerf.train()
    meta_nerf, meta_optim, meta_scaler = build_meta(nerf, optim, scaler)
    meta_state = MetaStateHolder(nerf, meta_nerf, meta_optim, meta_scaler)

    total_loss = 0.
    total_psnr = 0.
    batches = tqdm(loader, desc=f"[NeRF] Meta Initialization", disable=not verbose)
    for C, ro, rd in batches:
        C, ro, rd = C.to(d), ro.to(d), rd.to(d)
        
        for _ in range(meta_steps):
            with autocast(enabled=scaler.is_enabled()):
                C_ = raymarcher.render_volume(meta_nerf, ro, rd, perturb=perturb)
                meta_loss = criterion(C_, C)

            meta_scaler.scale(meta_loss).backward()
            meta_scaler.step(meta_optim)
            meta_scaler.update()
            meta_optim.zero_grad(set_to_none=True)

        with torch.no_grad():
            for p, meta_p in zip(nerf.parameters(), meta_nerf.parameters()):
                p.grad = scaler.scale(p - meta_p)

            with autocast(enabled=scaler.is_enabled()):
                C_ = raymarcher.render_volume(nerf, ro, rd, perturb=perturb)
                loss = criterion(C_, C)
                psnr = -10. * torch.log10(loss)
        
        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)
        meta_state.update()

        total_loss += loss.item() / len(loader)
        total_psnr += psnr.item() / len(loader)
        batches.set_postfix(loss=total_loss, psnr=total_psnr)
        
    del meta_nerf
    del meta_optim
    del meta_scaler
    
    torch.cuda.empty_cache()
    gc.collect()

    return total_loss, total_psnr