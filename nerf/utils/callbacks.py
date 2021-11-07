import matplotlib.pyplot as plt
import numpy as np

from torch import Tensor
from nerf.core.model import NeRF
from nerf.core.renderer import BoundedVolumeRaymarcher as BVR
from nerf.infer import infer
from nerf.utils.history import History
from PIL import Image


def render_callback(
    coarse: NeRF,
    fine: NeRF,
    raymarcher: BVR,
    ro: Tensor,
    rd: Tensor,
    C: Tensor,
    H: int,
    W: int,
    batch_size: int,
    path: str,
) -> None:
    """Render Callback
    
    Arguments:
        coarse (NeRF): coarse neural radiance field model
        fine (NeRF): fine neural radiance field model
        raymarcher (BoundedVolumeRaymarcher): bounded volume raymarcher renderer
        ro (Tensor): ray origins (B, 3)
        rd (Tensor): ray directions (B, 3)
        C (Tensor): ground truth colors (B, 3)
        H (int): height
        W (int): width
        batch_size (int): batch size
        path (str): path where to save the render
    """
    ro, rd = ro[:H * W], rd[:H * W]
    depth, rgb = infer(coarse, fine, raymarcher, ro, rd, H, W, batch_size=batch_size)
    depth, rgb = depth.numpy().astype(np.uint8), rgb.numpy().astype(np.uint8)
    gt = (C[:H * W].view(H, W, 3) * 255).numpy().astype(np.uint8)
    Image.fromarray(np.hstack((gt, rgb, depth))).save(path)


def plot_train_callback(history: History, scene: str, path: str) -> None:
    """Plot Train Callback

    Arguments:
        history (History): history
        scene (str): blender scene name
        path (str): path where to save the plot
    """
    fig = plt.figure(figsize=(12, 4))

    ax_mse = fig.add_subplot(131)
    ax_psnr = fig.add_subplot(132)
    ax_lr = fig.add_subplot(133)

    ax_mse.set_title(f"NeRF {scene} MSE")
    ax_mse.set_ylabel(f"MSE")
    ax_mse.set_xlabel("epochs")

    ax_psnr.set_title(f"NeRF {scene} PSNR")
    ax_psnr.set_ylabel(f"PSNR")
    ax_psnr.set_xlabel("epochs")
    
    ax_lr.set_title(f"NeRF {scene} Learning Rate")
    ax_lr.set_ylabel(f"LR")
    ax_lr.set_xlabel("steps")
    
    x = np.arange(1, len(history.train) + 1)
    
    if x.min() < x.max():
        ax_mse.set_xlim((x.min(), x.max()))
        ax_psnr.set_xlim((x.min(), x.max()))
    
    if len(history.train):
        mse = np.array([datum[0] for datum in history.train])
        psnr = np.array([datum[1] for datum in history.train])
        
        ax_mse.plot(x, mse, label="train")
        ax_psnr.plot(x, psnr, label="train")
        
    if len(history.val):
        mse = np.array([datum[0] for datum in history.val])
        psnr = np.array([datum[1] for datum in history.val])
        
        ax_mse.plot(x, mse, label="val")
        ax_psnr.plot(x, psnr, label="val")
    
    if history.test:
        mse = np.array([history.test[0]] * len(history.train))
        psnr = np.array([history.test[1]] * len(history.train))

        ax_mse.plot(x, mse, label="test")
        ax_psnr.plot(x, psnr, label="test")
        
    if len(history.lr):
        x = np.arange(1, len(history.lr) + 1)
        lr = np.array(history.lr)
        
        ax_lr.plot(x, lr)
        ax_lr.set_xlim((x.min(), x.max()))

    ax_lr.set_yscale("log")
    ax_lr.set_xscale("log")

    ax_mse.grid(linestyle="dotted")
    ax_psnr.grid(linestyle="dotted")
    ax_lr.grid(linestyle="dotted")
    
    ax_mse.legend()
    ax_psnr.legend()

    fig.tight_layout()
    fig.canvas.draw()
    fig.savefig(path)

    plt.close(fig)


def plot_distill_callback(history: History, scene: str, path: str) -> None:
    """Plot Distill Callback

    Arguments:
        history (History): history
        scene (str): blender scene name
        path (str): path where to save the plot
    """
    fig = plt.figure(figsize=(4 * 2, 4))
            
    ax_mse  = fig.add_subplot(121)
    ax_lr = fig.add_subplot(122)

    ax_mse.set_title(f"NeRF Distill {scene} MSE")
    ax_mse.set_ylabel(f"MSE")
    ax_mse.set_xlabel("steps")
    
    ax_lr.set_title(f"NeRF Distill {scene} Learning Rate")
    ax_lr.set_ylabel(f"LR")
    ax_lr.set_xlabel("steps")
        
    if len(history.train):
        x = np.arange(1, len(history.train) + 1)
        mse = np.array([datum[0] for datum in history.train])
        avg_mse = np.convolve(mse, np.ones(50) / 50., "valid")

        ax_mse.plot(x, mse, label="raw train")
        ax_mse.plot(avg_mse[:x.shape[0]], label="mavg train")

        if x.min() < x.max():
            ax_mse.set_xlim((x.min(), x.max()))
        
    if len(history.lr):
        x = np.arange(1, len(history.lr) + 1)
        lr = np.array(history.lr)
        
        ax_lr.plot(x, lr)
        if x.min() < x.max():
            ax_lr.set_xlim((x.min(), x.max()))

    ax_mse.set_yscale("log")
    ax_mse.set_xscale("log")

    ax_lr.set_yscale("log")
    ax_lr.set_xscale("log")

    ax_mse.grid(linestyle="dotted")
    ax_lr.grid(linestyle="dotted")
    
    ax_mse.legend()
    
    fig.tight_layout()
    fig.canvas.draw()
    fig.savefig(path)

    plt.close(fig)