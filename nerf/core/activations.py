import torch
import torch.jit as jit

from torch import Tensor
from torch.nn import Module


@jit.script
def widened_sigmoid(x: Tensor) -> Tensor:
    """Functional widened sigmoid activation
    
    Arguments:
        x (Tensor): input tensor

    Returns:
        x (Tensor): activated output tensor
    """
    EPS = 1e-3
        
    num = 1 + 2 * EPS
    den = 1 + torch.exp(-x)
    return num / den - EPS


@jit.script
def shifted_softplus(x: Tensor) -> Tensor:
    """Functional shifted softplus activation
    
    Arguments:
        x (Tensor): input tensor

    Returns:
        x (Tensor): activated output tensor
    """
    return torch.log(1 + torch.exp(x - 1))


class WidenedSigmoid(Module):
    """Module widened sigmoid activation"""
    
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward widened sigmoid activation
    
        Arguments:
            x (Tensor): input tensor

        Returns:
            x (Tensor): activated output tensor
        """
        return widened_sigmoid(x)


class ShiftedSoftplus(Module):
    """Module shifted softplus activation"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward shifted softplus activation
    
        Arguments:
            x (Tensor): input tensor

        Returns:
            x (Tensor): activated output tensor
        """
        return shifted_softplus(x)