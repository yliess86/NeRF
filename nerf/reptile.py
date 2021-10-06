import gc
import torch

from copy import deepcopy
from nerf.core.model import NeRF
from nerf.core.renderer import BoundedVolumeRaymarcher as BVR
from nerf.data.utils import loaders
from nerf.train import step
from nerf.utils.history import History
from nerf.utils.pbar import tqdm
from torch import device
from torch.cuda.amp import autocast, GradScaler
from torch.nn import Module
from torch.optim import Optimizer, SGD
from torch.utils.data import DataLoader, Dataset
from typing import Callable, List, Optional, Tuple


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


def meta_step(
    epoch: int,
    nerf: NeRF,
    raymarcher: BVR,
    optim: Optimizer,
    criterion: Module,
    scaler: GradScaler,
    loader: DataLoader,
    d: device,
    steps: Optional[int] = 16,
    perturb: Optional[bool] = False,
    verbose: Optional[bool] = True,
) -> Tuple[float, float]:
    """Meta Step

    Arguments:
        epoch (int): current meta epoch
        nerf (NeRF): neural radiance field model to be trained
        raymarcher (BVR): bounded volume raymarching renderer
        optim (Optimizer): optimization strategy
        criterion (Module): objective function
        scaler (GradScaler): grad scaler for half precision (fp16)
        loader (DataLoader): batch data loader
        d (device): torch device to send the batch on
        steps (Optional[int]): number of meta steps to perform before updating (default: 16)
        perturb (Optional[bool]): peturb ray query segment (default: False)
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
    batches = tqdm(loader, desc=f"[NeRF] Meta {epoch}", disable=not verbose)
    for C, ro, rd in batches:
        C, ro, rd = C.to(d), ro.to(d), rd.to(d)
        
        for _ in range(steps):
            with autocast(enabled=scaler.is_enabled()):
                _, C_ = raymarcher.render_volume(meta_nerf, ro, rd, perturb=perturb, train=True)
                meta_loss = criterion(C_, C)

            meta_scaler.scale(meta_loss).backward()
            meta_scaler.step(meta_optim)
            meta_scaler.update()
            meta_optim.zero_grad(set_to_none=True)

        with torch.no_grad():
            for p, meta_p in zip(nerf.parameters(), meta_nerf.parameters()):
                p.grad = scaler.scale(p - meta_p)

            with autocast(enabled=scaler.is_enabled()):
                _, C_ = raymarcher.render_volume(nerf, ro, rd, perturb=perturb, train=False)
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


def reptile_fit(
    nerf: NeRF,
    raymarcher: BVR,
    optim: Optimizer,
    criterion: Module,
    scaler: GradScaler,
    train_data: Dataset,
    val_data: Optional[Dataset] = None,
    test_data: Optional[Dataset] = None,
    epochs: Optional[int] = 100,
    steps: Optional[int] = 16,
    batch_size: Optional[int] = 32,
    jobs: Optional[int] = 8,
    perturb: Optional[bool] = False,
    callbacks: Optional[List[Callable[[int, History], None]]] = [],
    verbose: bool = True,
) -> History:
    """Reptile Fit NeRF on a specific dataset

    Arguments:
        nerf (NeRF): neural radiance field model to be trained
        raymarcher (BVR): bounded volume raymarching renderer
        optim (Optimizer): optimization strategy
        criterion (Module): objective function
        scaler (GradScaler): grad scaler for half precision (fp16)
        train_data (Dataset): training dataset
        val_data (Optional[Dataset]): validation dataset (default: None)
        test_data (Optional[Dataset]): testing dataset (default: None)
        epochs (Optional[int]): amount of epochs to train (default: 100)
        steps (Optional[int]): meta steps if meta learning (default: 16)
        batch_size (Optional[int]): batch size (default: 32)
        jobs (Optional[int]): number of processes to use  (default: 8)
        perturb (Optional[bool]): peturb ray query segment (default: False)
        callbacks (Optional[List[Callable[[int, History], None]]]): callbacks (default: [])
        verbose (Optional[bool]): print tqdm (default: True)

    Returns:
        history (History): training history
    """
    datasets = train_data, val_data, test_data
    train, val, test = loaders(*datasets, batch_size, jobs)

    H = History()
    d = next(nerf.parameters()).device
    args = nerf, raymarcher, optim, criterion, scaler

    meta_opt = { "steps": steps, "perturb": perturb, "verbose": verbose }
    val_opt = { "split": "val", "verbose": verbose }
    test_opt = { "split": "test", "verbose": verbose }

    pbar = tqdm(range(epochs), desc="[NeRF] Epoch", disable=not verbose)
    for epoch in pbar:
        H.train.append(meta_step(epoch, *args, train, d, **meta_opt))
        pbar.set_postfix(mse=H.train[-1][0], psnr=H.train[-1][1])
        
        if val:
            H.val.append(step(epoch, *args, val, d, **val_opt))
            pbar.set_postfix(mse=H.val[-1][0], psnr=H.val[-1][1])
        
        for callback in callbacks:
            callback(epoch, H)
    
    if test:
        H.test = step(epoch, *args, test, d, **test_opt)
        pbar.set_postfix(mse=H.test[0], psnr=H.test[1])

        for callback in callbacks:
            callback(epoch, H)

    return H


NeRF.reptile_fit = reptile_fit