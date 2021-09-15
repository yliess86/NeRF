import torch
import torch.jit as jit

from numpy import pi as PI
from torch import Size, Tensor
from torch.nn import Module, Parameter


def sample_b(size: Size, sigma: float) -> Tensor:
    return torch.randn(size) * sigma


@jit.scipt
def map_fourier_features(v: Tensor, b: Tensor) -> Tensor:
    a = 2 * PI * v @ b.T
    return torch.cat((torch.sin(a), torch.cos(a)), dim=-1)


class FourierFeatures(Module):
    def __init__(
        self,
        i_dim: int,
        features: int = 256,
        sigma: float = 6.,
    ) -> None:
        super().__init__()
        self.i_dim = i_dim
        self.features = features
        self.sigma = sigma

        self.b = sample_b(Size((self.features, self.i_dim)), self.sigma)
        self.b = Parameter(self.b, requires_grad=False)

    def forward(self, v: Tensor) -> Tensor:
        return map_fourier_features(v, self.b)