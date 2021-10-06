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
        [torch.cos(theta), 0., -torch.sin(theta), 0.],
        [              0., 1.,                0., 0.],
        [torch.sin(theta), 0.,  torch.cos(theta), 1.],
        [              0., 0.,                0., 1.],
    ])


@jit.script
def turnaround(theta: float, phi: float, radius: float) -> Tensor:
    """Turnaround matrix (phi, theta, radius)

    Arguments:
        theta (float): angle theta
        phi (float): angle phi
        z (float): depth z

    Returns:
        M (Tensor): turnaround matrix (4, 4)
    """
    rt = rotation_theta(theta)
    rp = rotation_phi(phi)
    tz = translation_z(radius)

    return tensor([
        [-1., 0., 0., 0.],
        [ 0., 1., 0., 0.],
        [ 0., 0., 1., 0.],
        [ 0., 0., 0., 1.],
    ]) @ rt @ rp @ tz


@jit.script
def turnaround_poses(
    theta: Tuple[float, float],
    phi: Tuple[float, float],
    radius: float,
    samples: int,
) -> Tensor:
    """Turnaround matrices (phi, theta, radius)

    Arguments:
        theta (Tuple[float, float]): angle range theta
        phi (Tuple[float, float]): angle range phi
        z (float): depth z
        samples (int): number of sample N along the path

    Returns:
        poses (Tensor): turnaround theta matrices (N, 4, 4)
    """
    ts = torch.linspace(*theta, samples + 1)[:-1]
    ps = torch.linspace(*phi, samples + 1)[:-1]
    pos = [turnaround(*tp, radius) for tp in zip(ts, ps)]
    return torch.stack(pos, dim=0)