import numpy as np
import torch
import os
import json

from nerf.core.ray import pinhole_ray_directions, phinhole_ray_projection
from PIL import Image
from torch import FloatTensor, Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from tqdm import tqdm
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
    return  .5 * W / np.tan(.5 * camera_angle_x)


def read_data(
    dataset: "BlenderDataset",
    base_dir: str,
    meta: Dict[str, Any],
    step: int,
) -> Tuple[Tensor, Tensor]:
    """Extract dataset data from metadata

    Arguments:
        dataset (BlenderDataset): dataset context
        base_dir (str): data directory
        meta (Dict[str, Any]): dataset metadata
        step (int): read every x file

    Returns:
        imgs (Tensor): view images (N, W, H, 3)
        poses (Tensor): camera to world matrices (N, 4, 4)
    """
    to_tensor = ToTensor()
    frames = meta["frames"][::step]
    
    imgs, poses = [], []
    for frame in tqdm(frames, desc=f"[{str(dataset)}] Loading Data"):
        img = os.path.join(base_dir, f"{frame['file_path']}.png")
        img = Image.open(img)
        img = to_tensor(img).float().permute(1, 2, 0)
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
    W: int,
    H: int,
    focal: float,
    poses: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Generate dataset rays (origin and direction)

    Arguments:
        dataset (BlenderDataset): dataset context
        W (int): frame width
        H (int): frame height
        focal (float): camera focal length
        poses (Tensor): camera to world matrices (N, 4, 4)

    Returns:
        ro (Tensor): ray origins (N, W, H, 3)
        rd (Tensor): ray directions (N, W, H, 3)
    """
    prd = pinhole_ray_directions(W, H, focal)

    ros, rds = [], []
    for c2w in tqdm(poses, desc=f"[{str(dataset)}] Building Rays"):
        ro, rd = phinhole_ray_projection(prd, c2w)
        ros.append(ro)
        rds.append(rd)
    
    return torch.stack(ros, dim=0), torch.stack(rds, dim=0)


class BlenderDataset(Dataset):
    """Blender Synthetic NeRF Dataset
    
    Arguments:
        root (str): dataset directory
        scene (str): blender scene
        split (int): dataset split ("train", "val", "test")
        step (int): read every x frame (default: 1)
    """

    def __init__(
        self,
        root: str,
        scene: str,
        split: str,
        step: int = 1,
    ) -> None:
        super().__init__()
        self.root = root
        self.scene = scene
        self.split = split
        self.step = step

        self.base_dir = os.path.join(self.root, self.scene)
        self.meta = read_meta(self.base_dir, self.split)
        
        self.imgs, self.poses = read_data(
            self, self.base_dir, self.meta, self.step,
        )
        
        self.SIZE = self.W, self.H = self.imgs[0].shape[:2][::-1]
        self.focal = read_focal(self.W, self.meta)
        self.near, self.far = 2., 6.
        
        self.ro, self.rd = build_rays(
            self, *self.SIZE, self.focal, self.poses,
        )

        self.C = self.imgs.view(-1, 3)
        self.ro = self.ro.view(-1, 3)
        self.rd = self.rd.view(-1, 3)

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