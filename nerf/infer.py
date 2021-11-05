"""python3 -m nerf.infer

Render Turnaround a GIF using NeRF.
"""
import torch

from nerf.core.model import NeRF
from nerf.core.renderer import BoundedVolumeRaymarcher as BVR
from nerf.utils.pbar import tqdm
from torch import Tensor


@torch.inference_mode()
def infer(
    nerf: NeRF,
    raymarcher: BVR,
    ro: Tensor,
    rd: Tensor,
    H: int,
    W: int,
    batch_size: int = 32,
    verbose: bool = True,
) -> Tensor:
    """Neural radiance field inference (render frame)

    Arguments:
        nerf (NeRF): neural radiance field model
        raymarcher (BVR): bounded volume raymarcher
        ro (Tensor): rays origin (B, 3)
        rd (Tensor): rays direction (B, 3)
        H (int): frame height
        W (int): frame width
        batch_size (int): batch size
        verbose (bool): print tqdm

    Returns:
        depth_map (Tensor): rendered depth map [0, 255] (H, W, 3)
        rgb_map (Tensor): rendered rgb map [0, 255] (H, W, 3)
    """
    d = next(nerf.parameters()).device
    n = ro.size(0)
    
    ro, rd = ro.to(d), rd.to(d)

    depth_map = torch.zeros((H * W, ), dtype=torch.float32)
    rgb_map = torch.zeros((H * W, 3), dtype=torch.float32)

    batches = range(0, n, batch_size)
    for s in tqdm(batches, desc="[NeRF] Rendering", disable=not verbose):
        e = min(s + batch_size, n)
        *_, D, C = raymarcher.render_volume(nerf, ro[s:e], rd[s:e], train=False)
        
        depth_map[s:e] = D.cpu()
        rgb_map[s:e] = C.cpu()

    depth_map = depth_map.view(H, W, 1)
    depth_map = depth_map.repeat((1, 1, 3))
    depth_map = depth_map.nan_to_num()
    depth_map = depth_map - depth_map.min()
    depth_map = depth_map / (depth_map.max() + 1e-10)
    depth_map = depth_map.clip(0, 1) * 255

    rgb_map = rgb_map.view(H, W, 3)
    rgb_map = rgb_map.clip(0, 1) * 255
    
    return depth_map, rgb_map


NeRF.infer = infer


if __name__ == "__main__":
    import numpy as np
    import torch.jit as jit

    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    from moviepy.editor import ImageSequenceClip
    from nerf.core.ray import pinhole_ray_directions, phinhole_ray_projection
    from nerf.data.path import turnaround_poses
    from torch.cuda.amp import autocast


    CAM_ANGLE_X = 0.6194058656692505


    parser = ArgumentParser(__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input",       type=str,   required=True,       help="TorchScript NeRF path")
    parser.add_argument(      "--height",      type=int,   default=800,         help="Frame height")
    parser.add_argument(      "--width",       type=int,   default=800,         help="Frame width")
    parser.add_argument(      "--near",        type=float, default=2.,          help="Near Plane")
    parser.add_argument(      "--far",         type=float, default=6.,          help="Far Plane")
    parser.add_argument(      "--cam_angle_x", type=float, default=CAM_ANGLE_X, help="Camera angle x to compute the camera focal length")
    parser.add_argument("-c", "--coarse",      type=int,   default=64,          help="Coarse samples")
    parser.add_argument("-f", "--fine",        type=int,   default=64,          help="Fine samples")
    parser.add_argument("-b", "--batch_size",  type=int,   default=16_384,      help="Batch size")
    parser.add_argument(      "--frames",      type=int,   default=100,         help="Number of frames to render")
    parser.add_argument(      "--fps",         type=int,   default=15,          help="FPS to render")
    parser.add_argument("-d", "--device",      type=int,   default=0,           help="Cuda GPU ID (-1 for CPU)")
    parser.add_argument(      "--amp",                     action="store_true", help="Automatic Mixted Precision")
    args = parser.parse_args()

    device = "cpu" if args.device < 0 else f"cuda:{args.device}"
    nerf = jit.load(args.input).to(device)
    raymarcher = BVR(args.near, args.far, args.coarse, args.fine)
     
    ranges = (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi), (4., 4.)
    poses = turnaround_poses(*ranges, args.frames)

    focal = .5 * args.width / np.tan(.5 * args.cam_angle_x)
    prd = pinhole_ray_directions(args.height, args.width, focal)

    ros = torch.zeros((poses.size(0), args.height, args.width, 3), dtype=torch.float32)
    rds = torch.zeros((poses.size(0), args.height, args.width, 3), dtype=torch.float32)
    for i, pose in enumerate(poses):
        ros[i], rds[i] = phinhole_ray_projection(prd, pose)
    ros = ros.view(-1, 3).to(device)
    rds = rds.view(-1, 3).to(device)
    
    depth_maps = np.zeros((args.frames, args.height, args.width, 3), dtype=np.uint8)
    rgb_maps = np.zeros((args.frames, args.height, args.width, 3), dtype=np.uint8)

    depth_path = args.input.replace(".model.ts", ".depth.gif")
    rgb_path = args.input.replace(".model.ts", ".rgb.gif")

    S = args.height * args.width
    frames = tqdm(range(0, args.frames * S, S), desc="[NeRF] Rendering Frame")
    for i, s in enumerate(frames):
        with autocast(enabled=args.amp):
            depth_map, rgb_map = infer(
                nerf,
                raymarcher,
                ros[s:s + S],
                rds[s:s + S],
                args.height,
                args.width,
                args.batch_size,
                verbose=False,
            )

        depth_maps[i] = depth_map.numpy().astype(np.uint8)
        rgb_maps[i] = rgb_map.numpy().astype(np.uint8)

    durations = [1. / args.fps] * args.frames
    ImageSequenceClip(list(depth_maps), durations=durations).write_gif(depth_path, fps=args.fps)
    ImageSequenceClip(list(rgb_maps), durations=durations).write_gif(rgb_path, fps=args.fps)