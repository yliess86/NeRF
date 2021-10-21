import numpy as np
import torch

from nerf.core.model import NeRF
from nerf.core.renderer import BoundedVolumeRaymarcher as BVR
from nerf.core.scheduler import Scheduler
from nerf.data.utils import loaders
from nerf.utils.history import History
from nerf.utils.pbar import tqdm
from torch import device
from torch.cuda.amp import autocast, GradScaler
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from typing import Callable, List, Optional, Tuple


GRAD_NORM_CLIP = 0.


def step(
    epoch: int,
    nerf: NeRF,
    raymarcher: BVR,
    optim: Optimizer,
    scheduler: Scheduler,
    criterion: Module,
    scaler: GradScaler,
    loader: DataLoader,
    d: device,
    split: str,
    perturb: Optional[bool] = False,
    verbose: Optional[bool] = True,
) -> Tuple[List[float], List[float], List[float]]:
    """Training/Validation/Testing step

    Arguments:
        epoch (int): current epoch
        nerf (NeRF): neural radiance field model to be trained
        raymarcher (BVR): bounded volume raymarching renderer
        optim (Optimizer): optimization strategy
        scheduler (Scheduler): learning rate scheduler
        criterion (Module): objective function
        scaler (GradScaler): grad scaler for half precision (fp16)
        loader (DataLoader): batch data loader
        d (device): torch device to send the batch on
        split (str): step state ("train", "val", "test")
        perturb (Optional[bool]): peturb ray query segment (default: False)
        verbose (Optional[bool]): print tqdm (default: True)

    Returns:
        steps_loss (List[float]): loss
        steps_psnr (List[float]): psnr
        steps_lr (List[float]): learning rate
    """
    train = split == "train"
    nerf = nerf.train(train)
    
    total_loss, total_psnr, total_lr = 0., 0., 0.
    steps_loss, steps_psnr, steps_lr = [], [], []

    desc = f"[NeRF] {split.capitalize()} {epoch + 1}"
    batches = tqdm(loader, desc=desc, disable=not verbose)

    with torch.set_grad_enabled(train):
        for C, ro, rd in batches:
            C, ro, rd = C.to(d), ro.to(d), rd.to(d)

            with autocast(enabled=scaler.is_enabled()):
                *_, C_ = raymarcher.render_volume(nerf, ro, rd, perturb=perturb, train=train)
                loss = criterion(C_, C)
            
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                if GRAD_NORM_CLIP > 0.: clip_grad_norm_(nerf.parameters(), GRAD_NORM_CLIP)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                scheduler.step()
                
                total_lr += scheduler.lr / len(loader)
                steps_lr.append(scheduler.lr)

            with torch.no_grad():
                psnr = -10. * torch.log10(loss)
            
            total_loss += loss.item() / len(loader)
            total_psnr += psnr.item() / len(loader)
            steps_loss.append(loss.item())
            steps_psnr.append(psnr.item())

            batches.set_postfix(loss=total_loss, psnr=total_psnr, lr=total_lr)

    return steps_loss, steps_psnr, steps_lr


def fit(
    nerf: NeRF,
    raymarcher: BVR,
    optim: Optimizer,
    scheduler: Scheduler,
    criterion: Module,
    scaler: GradScaler,
    train_data: Dataset,
    val_data: Optional[Dataset] = None,
    test_data: Optional[Dataset] = None,
    epochs: Optional[int] = 100,
    batch_size: Optional[int] = 32,
    jobs: Optional[int] = 8,
    perturb: Optional[bool] = False,
    callbacks: Optional[List[Callable[[int, History], None]]] = [],
    verbose: bool = True,
) -> History:
    """Fit NeRF on a specific dataset

    Arguments:
        nerf (NeRF): neural radiance field model to be trained
        raymarcher (BVR): bounded volume raymarching renderer
        optim (Optimizer): optimization strategy
        scheduler (Scheduler): learning rate scheduler
        criterion (Module): objective function
        scaler (GradScaler): grad scaler for half precision (fp16)
        train_data (Dataset): training dataset
        val_data (Optional[Dataset]): validation dataset (default: None)
        test_data (Optional[Dataset]): testing dataset (default: None)
        epochs (Optional[int]): amount of epochs to train (default: 100)
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
    args = nerf, raymarcher, optim, scheduler, criterion, scaler

    train_opt = { "split": "train", "perturb": perturb, "verbose": verbose }
    val_opt = { "split": "val", "verbose": verbose }
    test_opt = { "split": "test", "verbose": verbose }

    pbar = tqdm(range(epochs), desc="[NeRF] Epoch", disable=not verbose)
    for epoch in pbar:
        mse, psnr, lr = step(epoch, *args, train, d, **train_opt)
        H.train.append((np.average(mse), np.average(psnr)))
        H.lr += lr
        pbar.set_postfix(mse=np.average(mse), psnr=np.average(psnr))
        
        if val:
            mse, psnr, _ = step(epoch, *args, val, d, **val_opt)
            H.val.append((np.average(mse), np.average(psnr)))
            pbar.set_postfix(mse=np.average(mse), psnr=np.average(psnr))
        
        for callback in callbacks:
            callback(epoch, H)
    
    if test:
        mse, psnr, _ = step(epoch, *args, test, d, **test_opt)
        H.test = np.average(mse), np.average(psnr)
        pbar.set_postfix(mse=np.average(mse), psnr=np.average(psnr))

        for callback in callbacks:
            callback(epoch, H)

    return H


NeRF.fit = fit