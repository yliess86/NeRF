import torch

from dataclasses import dataclass, field
from nerf.core.model import NeRF
from nerf.core.renderer import BoundedVolumeRaymarcher as BVR
from torch import device
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Iterable, List, Tuple


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
    )

    val = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=jobs,
    )

    test = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=jobs,
    )
    
    return train, val, test


def step(
    nerf: NeRF,
    raymarcher: BVR,
    optim: Optimizer,
    criterion: Module,
    loader: DataLoader,
    d: device,
    split: str,
) -> Tuple[float, float]:
    """Training/Validation/Testing step

    Arguments:
        nerf (NeRF): neural radiance field model to be trained
        raymarcher (BVR): bounded volume raymarching renderer
        optim (Optimizer): optimization strategy
        criterion (Module): objective function
        loader (DataLoader): batch data loader
        d (device): torch device to send the batch on
        split (str): step state ("train", "val", "test")

    Returns:
        total_loss (float): arveraged cumulated total loss
        total_psnr (float): arveraged cumulated total psnr
    """
    train = split == "train"
    nerf = nerf.train() if train else nerf.eval()

    total_loss = 0.
    total_psnr = 0.
    batches = tqdm(loader, desc=f"[NeRF] {split.capitalize()}")
    for C, ro, rd in batches:
        C, ro, rd = C.to(d), ro.to(d), rd.to(d)

        C_ = raymarcher.render_volume(nerf, ro, rd)
        loss = criterion(C_, C)
        psnr = -10. * torch.log10(loss)

        if train:
            loss.backward()
            optim.step()
            optim.zero_grad()

        total_loss += loss.item() / len(loader)
        total_psnr += psnr.item() / len(loader)
        batches.set_postfix(loss=total_loss, psnr=total_psnr)

    return total_loss, total_psnr


def fit(
    nerf: NeRF,
    raymarcher: BVR,
    optim: Optimizer,
    criterion: Module,
    train_data: Dataset,
    val_data: Dataset,
    test_data: Dataset,
    epochs: int = 100,
    batch_size: int = 32,
    jobs: int = 8,
) -> History:
    """Fit NeRF on a specific dataset

    Arguments:
        nerf (NeRF): neural radiance field model to be trained
        raymarcher (BVR): bounded volume raymarching renderer
        optim (Optimizer): optimization strategy
        criterion (Module): objective function
        train_data (Dataset): training dataset
        val_data (Dataset): validation dataset
        test_data (Dataset): testing dataset
        epochs (int): amount of epochs to train
        batch_size (int): batch size
        jobs (int): number of processes to use

    Returns:
        history (History): training history
    """
    datasets = train_data, val_data, test_data
    train, val, test = loaders(*datasets, batch_size, jobs)

    d = next(nerf.parameters()).device
    modules = nerf, raymarcher, optim, criterion

    H = History()
    for _ in tqdm(range(epochs), desc="[NeRF] Epoch"):
        H.train.append(step(*modules, train, d, split="train"))
        H.val.append(step(*modules, val, d, split="val"))
    H.test = step(*modules, test, d, split="test")

    return H


NeRF.fit = fit