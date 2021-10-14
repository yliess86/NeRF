import torch

from nerf.core.features import FeatureMapping
from nerf.core.activations import ShiftedSoftplus, WidenedSigmoid
from torch import Tensor
from torch.nn import Linear, Module, ReLU, Sequential
from typing import Tuple


class NeRF(Module):
    """Neural Radiance Field (NeRF) module

    Arguments:
        phi_x (FeatureMapping): ray position embedder
        phi_d (FeatureMapping): ray direction embedder
        width (int): base number of neurons for each layer (default: 256)
        depth (int): number of layers in each subnetwork (default: 4)
        activ (Module): activation function used in hidden layers (default: ReLU)
    """
    
    def __init__(
        self,
        phi_x: FeatureMapping,
        phi_d: FeatureMapping,
        width: int = 256,
        depth: int = 8,
        activ: Module = ReLU,
    ) -> None:
        super().__init__()
        self.phi_x = phi_x
        self.phi_d = phi_d
        self.width = width
        self.depth = depth
        self.activ = activ

        hd = self.depth // 2
        fc = [(self.width, self.width) for _ in range(max(hd - 1, 0))]
        fc_1 = [(self.phi_x.o_dim             , self.width)] + fc
        fc_2 = [(self.phi_x.o_dim + self.width, self.width)] + fc

        self.fc_1 = Sequential(*(Sequential(Linear(*io), activ()) for io in fc_1))
        self.fc_2 = Sequential(*(Sequential(Linear(*io), activ()) for io in fc_2))
        
        self.sigma = Sequential(Linear(self.width, 1), ShiftedSoftplus())
        
        self.feature = Linear(self.width, self.width)
        self.rgb = Sequential(
            Linear(self.phi_d.o_dim + self.width, self.width // 2), activ(),
            Linear(self.width // 2, 3), WidenedSigmoid(),
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

    def requires_grad(self, required: bool = True) -> None:
        """Hot Fix for Potential Issue (TODO: Report on Forum)
        
        Arguments:
            required (bool): requires grad or not (default: True)
        """
        for p in self.parameters():
            p.requires_grad_(required)