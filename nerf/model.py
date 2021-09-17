import torch

from nerf.features import FourierFeatures
from torch import Tensor
from torch.nn import Linear, Module, ReLU, Sequential, Sigmoid
from typing import Tuple


class NeRF(Module):
    """Neural Radiance Field (NeRF) module

    Arguments:
        phi_x_dim (int): input features for ray position embedding
        phi_d_dim (int): input features for ray direction embedding
        width (int): base number of neurons for each layer (default: 256)
        depth (int): number of layers in each subnetwork (default: 4)
    """
    
    def __init__(
        self,
        phi_x_dim: int,
        phi_d_dim: int,
        width: int = 256,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.phi_x_dim = phi_x_dim
        self.phi_d_dim = phi_d_dim
        self.width = width
        self.depth = depth

        fc = [(self.width, self.width) for _ in range(self.depth - 1)]
        fc_1 = [(self.phi_x_dim * 2, self.width)] + fc
        fc_2 = [(self.phi_x_dim * 2 + self.width, self.width)] + fc

        self.phi_x = FourierFeatures(3, self.phi_x_dim)
        self.phi_d = FourierFeatures(3, self.phi_d_dim)

        self.fc_1 = Sequential(*(Sequential(Linear(*io), ReLU()) for io in fc_1))
        self.fc_2 = Sequential(*(Sequential(Linear(*io), ReLU()) for io in fc_2))
        
        self.sigma = Sequential(Linear(self.width, 1), ReLU())
        
        self.feature = Linear(self.width, self.width)
        self.rgb = Sequential(
            Linear(self.phi_d_dim  * 2 + self.width, self.width // 2), ReLU(),
            Linear(self.width // 2, 3), Sigmoid(),
        )

    def forward(self, rx: Tensor, rd: Tensor) -> Tuple[Tensor, Tensor]:
        """Query NeRF

        Arguments:
            rx (Tensor): ray query position (B, 3)
            rd (Tensor): ray query direction (B, 3)

        Returns:
            sigma (Tensor): volume density at query position (B, )
            rgb (Tensor): color at query position (B, 3)
        """
        phi_x = self.phi_x(rx)
        phi_d = self.phi_d(rd)

        x = self.fc_1(phi_x)
        x = self.fc_2(torch.cat((phi_x, x), dim=-1))

        sigma = self.sigma(x).unsqueeze(-1)
        rgb = self.rgb(torch.cat((phi_d, self.feature(x)), dim=-1))

        
        return sigma, rgb
        