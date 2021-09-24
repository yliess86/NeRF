import torch

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
    perturb: bool,
    verbose: bool,
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
        perturb (bool): peturb ray query segment
        verbose (bool): print tqdm

    Returns:
        total_loss (float): arveraged cumulated total loss
        total_psnr (float): arveraged cumulated total psnr
    """
    train = split == "train"
    nerf = nerf.train(train)

    total_loss = 0.
    total_psnr = 0.
    batches = tqdm(loader, desc=f"[NeRF] {split.capitalize()}", disable=not verbose)
    for C, ro, rd in batches:
        C, ro, rd = C.to(d), ro.to(d), rd.to(d)

        with autocast(enabled=scaler.is_enabled()):
            C_ = raymarcher.render_volume(nerf, ro, rd, perturb=perturb)
            loss = criterion(C_, C)
            psnr = -10. * torch.log10(loss)

        if train:
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

        total_loss += loss.item() / len(loader)
        total_psnr += psnr.item() / len(loader)
        batches.set_postfix(loss=total_loss, psnr=total_psnr)

    return total_loss, total_psnr


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
        val_data (Dataset): validation dataset (default: None)
        test_data (Dataset): testing dataset (default: None)
        epochs (int): amount of epochs to train (default: 100)
        batch_size (int): batch size (default: 32)
        jobs (int): number of processes to use  (default: 8)
        perturb (bool): peturb ray query segment (default: False)
        callbacks (Optional[List[Callable[[int, History], None]]]): callbacks (default: [])
        verbose (Optional[bool]): print tqdm (default: True)

    Returns:
        history (History): training history
    """
    datasets = train_data, val_data, test_data
    train, val, test = loaders(*datasets, batch_size, jobs)

    d = next(nerf.parameters()).device
    modules = nerf, raymarcher, optim, criterion, scaler

    H = History()
    pbar = tqdm(range(epochs), desc="[NeRF] Epoch", disable=not verbose)
    for epoch in pbar:
        H.train.append(step(*modules, train, d, split="train", perturb=perturb, verbose=verbose))
        pbar.set_postfix(mse=H.train[-1][0], psnr=H.train[-1][1])
        
        if val:
            H.val.append(step(*modules, val, d, split="val", verbose=verbose))
            pbar.set_postfix(mse=H.val[-1][0], psnr=H.val[-1][1])
        
        for callback in callbacks:
            callback(epoch, H)
    
    if test:
        H.test = step(*modules, test, d, split="test", verbose=verbose)
        pbar.set_postfix(mse=H.test[0], psnr=H.test[1])

    return H


NeRF.fit = fit