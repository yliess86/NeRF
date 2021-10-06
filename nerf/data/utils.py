from torch.utils.data import DataLoader, Dataset
from typing import Tuple


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
        pin_memory=True,
    ) if train_data else None

    val = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=jobs,
        pin_memory=True,
    ) if val_data else None

    test = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=jobs,
        pin_memory=True,
    ) if test_data else None
    
    return train, val, test