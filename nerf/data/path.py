import numpy as np
import torch
import torch.jit as jit

from torch import tensor, Tensor
from typing import Tuple


def translation_z(z: float) -> Tensor:
    """Translation matrix on Z-axis

    Arguments:
        z (float): depth z

    Returns:
        T (Tensor): transformation matrix for Z-axis translation (4, 4)
    """
    return tensor([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., z ],
        [0., 0., 0., 1.],
    ])


def rotation_psy(psy: float) -> Tensor:
    """Rotation matrix psy (yz plane)

    Arguments:
        psy (float): angle psy

    Returns:
        R (Tensor): transformation matrix for psy rotation (4, 4)
    """
    return tensor([
        [torch.cos(psy), -torch.sin(psy), 0., 0.],
        [torch.sin(psy),  torch.cos(psy), 0., 0.],
        [            0.,              0., 1., 0.],
        [            0.,              0., 0., 1.],
    ])


def rotation_phi(phi: float) -> Tensor:
    """Rotation matrix phi (xy plane)

    Arguments:
        phi (float): angle phi

    Returns:
        R (Tensor): transformation matrix for phi rotation (4, 4)
    """
    return tensor([
        [1.,              0.,              0., 0.],
        [0.,  torch.cos(phi), -torch.sin(phi), 0.],
        [0.,  torch.sin(phi),  torch.cos(phi), 0.],
        [0.,              0.,              0., 1.],
    ])


def rotation_theta(theta: float) -> Tensor:
    """Rotation matrix theta (zx plane)

    Arguments:
        theta (float): angle theta

    Returns:
        R (Tensor): transformation matrix for theta rotation (4, 4)
    """
    return tensor([
        [ torch.cos(theta), 0., torch.sin(theta), 0.],
        [               0., 1.,               0., 0.],
        [-torch.sin(theta), 0., torch.cos(theta), 0.],
        [               0., 0.,               0., 1.],
    ])


@jit.script
def turnaround(theta: float, phi: float, psy: float, radius: float) -> Tensor:
    """Turnaround matrix (phi, theta, radius)

    Arguments:
        theta (float): angle theta
        phi (float): angle phi
        psy (float): angle psy
        radius (float): distance from center of rotation

    Returns:
        M (Tensor): turnaround matrix (4, 4)
    """
    t = rotation_theta(theta)
    p = rotation_phi(phi)
    y = rotation_psy(psy)
    z = translation_z(radius)

    return tensor([
        [-1., 0., 0., 0.],
        [ 0., 1., 0., 0.],
        [ 0., 0., 1., 0.],
        [ 0., 0., 0., 1.],
    ]) @ t @ p @ y @ z


@jit.script
def turnaround_poses(
    theta: Tuple[float, float],
    phi: Tuple[float, float],
    psy: Tuple[float, float],
    radius: Tuple[float, float],
    samples: int,
) -> Tensor:
    """Turnaround matrices (psy, phi, theta, radius)

    Arguments:
        theta (Tuple[float, float]): angle range theta
        phi (Tuple[float, float]): angle range phi
        psy (Tuple[float, float]): angle range psy
        radius (Tuple[float, float]): depth range z (radius)
        samples (int): number of sample N along the path

    Returns:
        poses (Tensor): turnaround theta matrices (N, 4, 4)
    """
    ts = torch.linspace(*theta, samples + 1)[:-1]
    ps = torch.linspace(*phi, samples + 1)[:-1]
    ys = torch.linspace(*psy, samples + 1)[:-1]
    rs = torch.linspace(*radius, samples + 1)[:-1]
    pos = [turnaround(*tpyr) for tpyr in zip(ts, ps, ys, rs)]
    return torch.stack(pos, dim=0)