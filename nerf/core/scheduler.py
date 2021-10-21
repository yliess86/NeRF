import numpy as np

from torch.optim import Optimizer
from typing import Optional, Tuple


class Scheduler:
    """Learning Rate Scheduler

    Arguments:
        optim (Optimizer): optimizer to access learning rate
    """

    def __init__(self, optim: Optimizer) -> None:
        self.optim = optim
        self.current_step = 0.

    def step(self) -> "Scheduler":
        """Scheduler Step"""
        self.current_step += 1.
        self.update_optim()
        return self

    def update_optim(self) -> None:
        """Scheduler Optimizer Update"""
        for group in self.optim.param_groups:
            group["lr"] = self.lr

    @property
    def lr(self) -> float:
        """Current Scheduler Learning Rate"""
        raise NotImplementedError("Lr not implemented yet!")


class IndendityScheduler(Scheduler):
    """Identity Learning Rate Scheduler
    
    Arguments:
        optim (Optimizer): optimizer to access learning rate
        epochs (float): #epochs the model will be trained for
        lr (Optional[float]): learning rate (default: 5e-4)
    """

    def __init__(
        self,
        optim: Optimizer,
        epochs: float,
        lr: Optional[float] = 5e-4,
    ) -> None:
        super().__init__(optim)
        self._lr = lr
        self.update_optim()

    @property
    def lr(self) -> float:
        """Current Scheduler Learning Rate"""
        return self._lr


class LogDecayScheduler(Scheduler):
    """LogDecay Learning Rate Scheduler
    
    Arguments:
        optim (Optimizer): optimizer to access learning rate
        epochs (float): #epochs the model will be trained for
        steps_per_epoch (float): #batch per epoch (len(loader))
        lr_range (Optional[Tuple[float, float]]): min and max learning rate
            (default: (5e-6, 5e-4))
        power (float): inverse power of decay (default: 8)
    """

    def __init__(
        self,
        optim: Optimizer,
        epochs: float,
        steps_per_epoch: float,
        lr_range: Optional[Tuple[float, float]] = (5e-6, 5e-4),
        power: float = 8,
    ) -> None:
        super().__init__(optim)
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.lr_min, self.lr_max = lr_range
        self.power = power
        self.steps = self.epochs * steps_per_epoch

        self.update_optim()

    @property
    def lr(self) -> float:
        """Current Scheduler Learning Rate"""
        t = self.current_step / self.steps
        p = 1 - 10 ** (-self.power * (1 - t))
        return self.lr_min + (self.lr_max - self.lr_min) * p


class MipNeRFScheduler(Scheduler):
    """Mip-NeRF Learning Rate Scheduler
    
    Arguments:
        optim (Optimizer): optimizer to access learning rate
        epochs (float): #epochs the model will be trained for
        epochs_shift (float): #epochs to warmup
        steps_per_epoch (float): #batch per epoch (len(loader))
        lr_range (Optional[Tuple[float, float]]): min and max learning rate
            (default: (5e-6, 5e-4))
        scale (Optional[float]): scale factor for the warmup phase
    """

    def __init__(
        self,
        optim: Optimizer,
        epochs: float,
        epochs_shift: float,
        steps_per_epoch: float,
        lr_range: Optional[Tuple[float, float]] = (5e-6, 5e-4),
        scale: Optional[float] = 1e-2,
    ) -> None:
        super().__init__(optim)
        self.epochs = epochs
        self.epochs_shift = epochs_shift
        self.steps_per_epoch = steps_per_epoch
        self.lr_min, self.lr_max = lr_range
        self.scale = scale

        self.steps_shift = epochs_shift * steps_per_epoch
        self.steps = self.epochs * steps_per_epoch

        self.update_optim()

    @property
    def lr(self) -> float:
        """Current Scheduler Learning Rate"""
        t = np.clip(self.current_step / self.steps_shift, 0., 1.)
        scale = (1. - self.scale) * np.sin(.5 * np.pi * t) + self.scale * 1.
        
        t = self.current_step / self.steps
        lr = (1. - t) * np.log(self.lr_max) + t * np.log(self.lr_min)

        return scale * np.exp(lr)