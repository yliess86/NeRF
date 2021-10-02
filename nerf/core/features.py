import torch
import torch.jit as jit

from torch import Size, Tensor
from torch.nn import Module, Parameter


def sample_b(size: Size, sigma: float) -> Tensor:
    """Sample b matrix for fourier features
    
    Arguments:
        size (Size): b matrix size
        sigma (float): std of the gaussian

    Returns:
        b (Tensor): b matrix
    """
    return torch.randn(size) * sigma


@jit.script
def map_fourier_features(v: Tensor, b: Tensor) -> Tensor:
    """Map v to fourier features representation phi(v)
    
    Arguments:
        v (Tensor): input features (B, IFeatures)
        b (Tensor): b matrix (OFeatures, IFeatures)

    Returns:
        phi(v) (Tensor): fourrier features (B, 2 * OFeatures)
    """
    PI = 3.141592653589793
    a = 2 * PI * v @ b.T
    return torch.cat((torch.sin(a), torch.cos(a)), dim=-1)


class FourierFeatures(Module):
    """Fourier Features module

    Maps v to fourier features representation phi(v)

    Arguments:
        i_dim (int): input dimension for v
        features (int): output dimension (default: 256)
        sigma (float): std of the gaussian (default: 6.)
    """

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

        self.size = Size((self.features, self.i_dim))
        self.register_buffer("b", sample_b(self.size, self.sigma))

    def forward(self, v: Tensor) -> Tensor:
        """Map v to fourier features representation phi(v)
    
        Arguments:
            v (Tensor): input features (B, IFeatures)

        Returns:
            phi(v) (Tensor): fourrier features (B, 2 * OFeatures)
        """
        return map_fourier_features(v, self.b)