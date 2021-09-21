import numpy as np
import torch
import torch.jit as jit

from torch import tensor, Tensor


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
    """Rotation matrix phi

    Arguments:
        phi (float): angle phi

    Returns:
        R (Tensor): transformation matrix for phi rotation (4, 4)
    """
    return tensor([
        [1.,             0.,              0., 0.],
        [0., torch.cos(phi), -torch.sin(phi), 0.],
        [0., torch.sin(phi),  torch.cos(phi), 0.],
        [0.,             0.,              0., 1.],
    ])


def rotation_theta(theta: float) -> Tensor:
    """Rotation matrix theta

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
    c2w = translation_z(radius)
    c2w = rotation_phi(phi) @ c2w
    c2w = rotation_theta(theta) @ c2w
    return tensor([
        [-1., 0., 0., 0.],
        [ 0., 1., 0., 0.],
        [ 0., 0., 1., 0.],
        [ 0., 0., 0., 1.],
    ]) @ c2w


@jit.script
def turnaround_theta_poses(
    phi: float,
    radius: float,
    samples: int,
) -> Tensor:
    """Turnaround theta matrices (phi, theta, radius)

    Arguments:
        phi (float): angle phi
        z (float): depth z
        samples (int): number of sample N along the path

    Returns:
        poses (Tensor): turnaround theta matrices (N, 4, 4)
    """
    PI = 3.141592653589793
    thetas = torch.linspace(0, 2 * PI, samples + 1)[:-1]
    return torch.stack([
        turnaround(theta, phi, radius)
        for theta in thetas
    ], dim=0)


@jit.script
def turnaround_phi_poses(
    theta: float,
    radius: float,
    samples: int,
) -> Tensor:
    """Turnaround phi matrices (phi, theta, radius)

    Arguments:
        theta (float): angle theta
        z (float): depth z
        samples (int): number of sample N along the path

    Returns:
        poses (Tensor): turnaround phi matrices (N, 4, 4)
    """
    PI = 3.141592653589793
    phis = torch.linspace(0, 2 * PI, samples + 1)[:-1]
    return torch.stack([
        turnaround(theta, phi, radius)
        for phi in phis
    ], dim=0)