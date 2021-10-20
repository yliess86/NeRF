import torch
import torch.jit as jit

from torch import Size, Tensor
from torch.nn import Module


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
def map_positional_encoding(v: Tensor, freq_bands: Tensor) -> Tensor:
    """Map v to positional encoding representation phi(v)

    Arguments:
        v (Tensor): input features (B, IFeatures)
        freq_bands (Tensor): frequency bands (N_freqs, )

    Returns:
        phi(v) (Tensor): fourrier features (B, 3 + (2 * N_freqs) * 3)
    """
    pe = [v]
    for freq in freq_bands:
        fv = freq * v
        pe += [torch.sin(fv), torch.cos(fv)]
    return torch.cat(pe, dim=-1)


@jit.script
def map_fourier_features(v: Tensor, b: Tensor) -> Tensor:
    """Map v to fourier features representation phi(v)
    
    Arguments:
        v (Tensor): input features (B, IFeatures)
        b (Tensor): b matrix (OFeatures, IFeatures)

    Returns:
        phi(v) (Tensor): fourrier features (B, 2 * Features)
    """
    PI = 3.141592653589793
    a = 2 * PI * v @ b.T
    return torch.cat((torch.sin(a), torch.cos(a)), dim=-1)


class FeatureMapping(Module):
    """FeatureMapping Module
    
    Maps v to features following transformation phi(v)

    Arguments:
        i_dim (int): input dimensions
        o_dim (int): output dimensions
    """

    def __init__(self, i_dim: int, o_dim: int) -> None:
        super().__init__()
        self.i_dim = i_dim
        self.o_dim = o_dim

    def forward(self, v: Tensor) -> Tensor:
        """FeratureMapping forward pass
        
        Arguments:
            v (Tensor): input features (B, IFeatures)

        Returns:
            phi(v) (Tensor): mapped features (B, OFeatures)
        """
        raise NotImplementedError("Forward pass not implemented yet!")


class PositionalEncoding(FeatureMapping):
    """PositionalEncoding module

    Maps v to positional encoding representation phi(v)

    Arguments:
        i_dim (int): input dimension for v
        N_freqs (int): #frequency to sample (default: 10)
    """

    def __init__(
        self,
        i_dim: int,
        N_freqs: int = 10,
    ) -> None:
        super().__init__(i_dim, 3 + (2 * N_freqs) * 3)
        self.N_freqs = N_freqs

        a, b = 1, self.N_freqs - 1
        freq_bands = 2 ** torch.linspace(a, b, self.N_freqs)
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, v: Tensor) -> Tensor:
        """Map v to positional encoding representation phi(v)
    
        Arguments:
            v (Tensor): input features (B, IFeatures)

        Returns:
            phi(v) (Tensor): fourrier features (B, 3 + (2 * N_freqs) * 3)
        """
        return map_positional_encoding(v, self.freq_bands)


class FourierFeatures(FeatureMapping):
    """Fourier Features module

    Maps v to fourier features representation phi(v)

    Arguments:
        i_dim (int): input dimension for v
        features (int): output dimension (default: 256)
        sigma (float): std of the gaussian (default: 26.)
    """

    def __init__(
        self,
        i_dim: int,
        features: int = 256,
        sigma: float = 26.,
    ) -> None:
        super().__init__(i_dim, 2 * features)
        self.features = features
        self.sigma = sigma

        self.size = Size((self.features, self.i_dim))
        self.register_buffer("b", sample_b(self.size, self.sigma))

    def forward(self, v: Tensor) -> Tensor:
        """Map v to fourier features representation phi(v)
    
        Arguments:
            v (Tensor): input features (B, IFeatures)

        Returns:
            phi(v) (Tensor): fourrier features (B, 2 * Features)
        """
        return map_fourier_features(v, self.b)