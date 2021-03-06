import numpy as np
import torch
import os
import json

from nerf.core.ray import pinhole_ray_directions, phinhole_ray_projection
from nerf.data.path import turnaround_poses
from nerf.utils.pbar import tqdm
from PIL import Image
from torch import FloatTensor, Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from typing import Any, Dict, Tuple


def read_meta(base_dir: str, split: str) -> Dict[str, Any]:
    """Read dataset metadata

    Arguments:
        base_dir (str): data directory
        split (int): dataset split ("train", "val", "test")

    Returns:
        meta (Dict[str, Any]): dataset metadata
    """
    file = f"transforms_{split}.json"
    with open(os.path.join(base_dir, file), "r") as fp:
        meta = json.load(fp)
    return meta


def read_focal(W: int, meta: Dict[str, Any]) -> float:
    """Extract camera focal length from datset metadata

    Arguments:
        W (int): frame width
        meta (Dict[str, Any]): dataset metadata

    Returns:
        focal (float): camera focal length
    """
    camera_angle_x = float(meta["camera_angle_x"])
    return .5 * W / np.tan(.5 * camera_angle_x)


def read_data(
    dataset: "BlenderDataset",
    base_dir: str,
    meta: Dict[str, Any],
    step: int,
    scale: float,
) -> Tuple[Tensor, Tensor]:
    """Extract dataset data from metadata

    Arguments:
        dataset (BlenderDataset): dataset context
        base_dir (str): data directory
        meta (Dict[str, Any]): dataset metadata
        step (int): read every x file
            if step is `None` it will only read the first file
        scale (float): scale for smaller images

    Returns:
        imgs (Tensor): view images (N, H, W, 3)
        poses (Tensor): camera to world matrices (N, 4, 4)
    """
    to_tensor = ToTensor()
    frames = meta["frames"][::step] if step else meta["frames"][:1]
    
    imgs, poses = [], []
    for frame in tqdm(frames, desc=f"[{str(dataset)}] Loading Data"):
        img = os.path.join(base_dir, f"{frame['file_path']}.png")
        img = Image.open(img)

        if scale < 1.:
            w, h = img.width, img.height
            w = int(np.floor(scale * w))
            h = int(np.floor(scale * h))
            img = img.resize((w, h), Image.LANCZOS)

        img = to_tensor(img).permute(1, 2, 0)
        img = img[:, :, :3] * img[:, :, -1:]
        imgs.append(img)

        pose = frame["transform_matrix"]
        pose = FloatTensor(pose)
        poses.append(pose)

    imgs = torch.stack(imgs, dim=0)
    poses = torch.stack(poses, dim=0)
    
    return imgs, poses


def build_rays(
    dataset: "BlenderDataset",
    H: int,
    W: int,
    focal: float,
    poses: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Generate dataset rays (origin and direction)

    Arguments:
        dataset (BlenderDataset): dataset context
        H (int): frame height
        W (int): frame width
        focal (float): camera focal length
        poses (Tensor): camera to world matrices (N, 4, 4)

    Returns:
        ro (Tensor): ray origins (N, H, W, 3)
        rd (Tensor): ray directions (N, H, W, 3)
    """
    N = poses.size(0)

    prd = pinhole_ray_directions(H, W, focal)
    ros = torch.zeros((N, H, W, 3), dtype=torch.float32)
    rds = torch.zeros((N, H, W, 3), dtype=torch.float32)

    c2ws = tqdm(poses, desc=f"[{str(dataset)}] Building Rays")
    for i, c2w in enumerate(c2ws):
        ros[i], rds[i] = phinhole_ray_projection(prd, c2w)
    
    return ros, rds


class BlenderDataset(Dataset):
    """Blender Synthetic NeRF Dataset
    
    Arguments:
        root (str): dataset directory
        scene (str): blender scene
        split (int): dataset split ("train", "val", "test")
        step (int): read every x frame (default: 1)
            if step is `None` the dataset will just behave
            as a placeholder to create path given focal infos.
            It won't exhibit properties such as len nor getitem. 
        scale (float): scale for smaller images (default: 1.)
    """

    def __init__(
        self,
        root: str,
        scene: str,
        split: str,
        step: int = 1,
        scale: float = 1.,
    ) -> None:
        super().__init__()
        self.root = root
        self.scene = scene
        self.split = split
        self.step = step
        self.scale = max(min(scale, 1.), 0.)

        self.base_dir = os.path.join(self.root, self.scene)
        self.meta = read_meta(self.base_dir, self.split)
        
        self.imgs, self.poses = read_data(
            self,
            self.base_dir,
            self.meta,
            self.step,
            self.scale,
        )
        
        self.SIZE = self.H, self.W = self.imgs[0].shape[:2]
        self.focal = read_focal(self.W, self.meta)
        self.near, self.far = 2., 6.
        
        if step:
            self.ro, self.rd = build_rays(
                self,
                *self.SIZE,
                self.focal,
                self.poses,
            )

            self.C = self.imgs.view(-1, 3)
            self.ro = self.ro.view(-1, 3)
            self.rd = self.rd.view(-1, 3)

            assert self.C.size() == self.ro.size()
            assert self.C.size() == self.rd.size()

    def turnaround_data(
        self,
        theta: Tuple[float, float],
        phi: Tuple[float, float],
        psy: Tuple[float, float],
        radius: Tuple[float, float],
        samples: int = 40,
    ) -> Tuple[Tensor, Tensor]:
        """Turnaround data

        Arguments:
            theta (Tuple[float, float]): angle range theta
            phi (Tuple[float, float]): angle range phi
            psy (Tuple[float, float]): angle range psy
            radius (Tuple[float, float]): depth range z (radius)
            samples (int): number of sample N along the path (default: 40)

        Returns:
            ro (Tensor): ray origin (3, )
            rd (Tensor): ray direction (3, )
        """
        poses = turnaround_poses(theta, phi, psy, radius, samples)
        
        ro, rd  = build_rays(self, *self.SIZE, self.focal, poses)
        ro = ro.view(-1, 3)
        rd = rd.view(-1, 3)

        return ro, rd

    def __len__(self) -> int:
        """Dataset size
        
        Returns:
            len (int): dataset size
        """
        return self.C.size(0)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Retrieve data at given idx
        
        Arguments:
            idx (int): data index to retrieve

        Returns:
            C (Tensor): pixel color (3, )
            ro (Tensor): ray origin (3, )
            rd (Tensor): ray direction (3, )
        """
        return self.C[idx], self.ro[idx], self.rd[idx]

    def __str__(self) -> str:
        cls = self.__class__.__name__
        return f"{cls}({self.scene}, {self.split})"