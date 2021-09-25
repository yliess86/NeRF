import torch

from copy import deepcopy
from dataclasses import dataclass, field
from nerf.core.model import NeRF
from nerf.core.renderer import BoundedVolumeRaymarcher as BVR
from nerf.utils.pbar import tqdm
from torch import device
from torch.cuda.amp import autocast, GradScaler
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from typing import Callable, Iterable, List, Optional, Tuple


@dataclass
class History:
    """"Training history

    Arguments:
        train (List[Iterable[float]]): training history
        val (List[Iterable[float]]): validation history
        test (Iterable[float]): testing history
    """
    train: List[Iterable[float]] = field(default_factory=list)
    val: List[Iterable[float]] = field(default_factory=list)
    test: Iterable[float] = None


def loaders(
    train_data: Dataset,
    val_data: Dataset,
    test_data: Dataset,
    batch_size: int,
    jobs: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders

    Arguments:
        train_data (Dataset): training set
        val_data (Dataset): validation set
        test_data (Dataset): testing set
        batch_size (int): batch size
        jobs (int): number of processes to use

    Returns:
        train (DataLoader): training batch data loader
        val (DataLoader): validation batch data loader
        test (DataLoader): testing batch data loader
    """
    train = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=jobs,
    ) if train_data else None

    val = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=jobs,
    ) if val_data else None

    test = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=jobs,
    ) if test_data else None
    
    return train, val, test


def step(
    nerf: NeRF,
    raymarcher: BVR,
    optim: Optimizer,
    criterion: Module,
    scaler: GradScaler,
    loader: DataLoader,
    d: device,
    split: str,
    perturb: bool = False,
    meta_nerf: Optional[NeRF] = None,
    meta_optim: Optional[Optimizer] = None,
    meta_scaler: Optional[GradScaler] = None,
    meta_steps: Optional[int] = 16,
    verbose: Optional[bool] = True,
) -> Tuple[float, float]:
    """Training/Validation/Testing step

    Arguments:
        nerf (NeRF): neural radiance field model to be trained
        raymarcher (BVR): bounded volume raymarching renderer
        optim (Optimizer): optimization strategy
        criterion (Module): objective function
        scaler (GradScaler): grad scaler for half precision (fp16)
        loader (DataLoader): batch data loader
        d (device): torch device to send the batch on
        split (str): step state ("train", "val", "test")
        perturb (bool): peturb ray query segment (default: False)
        meta_nerf (NeRF): meta neural radiance field model to be trained (default: None)
        meta_optim (Optimizer): meta optimizer strategy (default: None)
        meta_scaler (GradScaler): meta grad scaler for half precision (fp16) (default: None)
        meta_steps (int): number of meta steps to perform before updating (default: 16)
        verbose (bool): print tqdm (default: True)

    Returns:
        total_loss (float): arveraged cumulated total loss
        total_psnr (float): arveraged cumulated total psnr
    """
    train = split == "train"
    nerf = nerf.train(train)

    meta = meta_nerf and meta_optim and train
    if meta: meta_nerf = meta_nerf.train()

    total_loss = 0.
    total_psnr = 0.
    batches = tqdm(loader, desc=f"[NeRF] {split.capitalize()}", disable=not verbose)
    for C, ro, rd in batches:
        C, ro, rd = C.to(d), ro.to(d), rd.to(d)

        if meta:
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
        
        if not meta: scaler.scale(loss).backward()
        if train:
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

        total_loss += loss.item() / len(loader)
        total_psnr += psnr.item() / len(loader)
        batches.set_postfix(loss=total_loss, psnr=total_psnr)

    return total_loss, total_psnr


def try_build_metas(
    nerf: NeRF,
    optim: Optimizer,
    scaler: GradScaler,
    meta: Optional[bool] = False,
) -> Tuple[NeRF, Optimizer, GradScaler]:
    """Try to build Meta Learning Objects

    Arguments:
        nerf (NeRF): neural radiance field model to be trained
        optim (Optimizer): optimization strategy
        scaler (GradScaler): grad scaler for half precision (fp16)
        meta (Optional[bool]): use meta learning (default: False)

    Returns:
        meta_nerf (NeRF): meta neural radiance field model to be trained (default: None)
        meta_optim (Optimizer): meta optimizer strategy (default: None)
        meta_scaler (GradScaler): meta grad scaler for half precision (fp16) (default: None)
    """
    optim_lr = optim.param_groups[0]["lr"]
    optim_cls = optim.__class__

    meta_nerf = deepcopy(nerf) if meta else None
    meta_optim = optim_cls(meta_nerf.parameters(), lr=optim_lr) if meta else None
    meta_scaler = deepcopy(scaler)

    return meta_nerf, meta_optim, meta_scaler


def fit(
    nerf: NeRF,
    raymarcher: BVR,
    optim: Optimizer,
    criterion: Module,
    scaler: GradScaler,
    train_data: Dataset,
    val_data: Optional[Dataset] = None,
    test_data: Optional[Dataset] = None,
    epochs: Optional[int] = 100,
    batch_size: Optional[int] = 32,
    jobs: Optional[int] = 8,
    perturb: Optional[bool] = False,
    meta: Optional[bool] = False,
    meta_steps: Optional[int] = 16,
    callbacks: Optional[List[Callable[[int, History], None]]] = [],
    verbose: bool = True,
) -> History:
    """Fit NeRF on a specific dataset

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
        batch_size (Optional[int]): batch size (default: 32)
        jobs (Optional[int]): number of processes to use  (default: 8)
        perturb (Optional[bool]): peturb ray query segment (default: False)
        meta (Optional[bool]): use meta learning (default: False)
        meta_steps (Optional[int]): meta steps if meta learning (default: 16)
        callbacks (Optional[List[Callable[[int, History], None]]]): callbacks (default: [])
        verbose (Optional[bool]): print tqdm (default: True)

    Returns:
        history (History): training history
    """
    datasets = train_data, val_data, test_data
    train, val, test = loaders(*datasets, batch_size, jobs)

    d = next(nerf.parameters()).device
    args = nerf, raymarcher, optim, criterion, scaler

    meta_nerf, meta_optim, meta_scaler = try_build_metas(nerf, optim, scaler, meta=meta)
    meta_args = {
        "meta_nerf": meta_nerf,
        "meta_optim": meta_optim,
        "meta_scaler": meta_scaler,
        "meta_steps": meta_steps,
    }
    
    train_opt = { "split": "train", "perturb": perturb, **meta_args, "verbose": verbose }
    val_opt = { "split": "val", "verbose": verbose }
    test_opt = { "split": "test", "verbose": verbose }

    H = History()
    pbar = tqdm(range(epochs), desc="[NeRF] Epoch", disable=not verbose)
    for epoch in pbar:
        H.train.append(step(*args, train, d, **train_opt))
        pbar.set_postfix(mse=H.train[-1][0], psnr=H.train[-1][1])
        
        if val:
            H.val.append(step(*args, val, d, **val_opt))
            pbar.set_postfix(mse=H.val[-1][0], psnr=H.val[-1][1])
        
        for callback in callbacks:
            callback(epoch, H)
    
    if test:
        H.test = step(*args, test, d, **test_opt)
        pbar.set_postfix(mse=H.test[0], psnr=H.test[1])

    return H


NeRF.fit = fit