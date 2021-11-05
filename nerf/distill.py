"""python3 -m nerf.distill

Design Student NeRF and Distill a Teacher NeRF into the Student.
"""
import numpy as np
import torch

from nerf.core.model import NeRF
from nerf.core.scheduler import Scheduler
from nerf.utils.history import History
from nerf.utils.pbar import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from typing import Callable, List, NamedTuple, Optional, Tuple


GRAD_NORM_CLIP = 0.


class Domain(NamedTuple):
    """Domain Infos"""
    x: Tuple[float, float]
    y: Tuple[float, float]
    z: Tuple[float, float]
    d: Tuple[float, float]


class VoxelSampler:
    """Voxel Sampler
    
    Arguments:
        domain (Domain): domain limits
        voxels (int): voxels per batch
        samples (int): samples per voxels
    """

    def __init__(self, domain: Domain, voxels: int, samples: int) -> None:
        self.domain = domain
        self.voxels = voxels
        self.samples = samples
        self.batch_size = self.voxels * self.samples

        self.voxel_size = torch.tensor([
            abs(domain.x[1] - domain.x[0]),
            abs(domain.y[1] - domain.y[0]),
            abs(domain.z[1] - domain.z[0]),
        ], dtype=torch.float32) / self.voxels
        
        linspace = lambda d: torch.linspace(*d, steps=self.voxels, dtype=torch.float32)
        x = linspace(domain.x)
        y = linspace(domain.y)
        z = linspace(domain.z)

        self.rx = torch.cartesian_prod(x, y, z)

    def __len__(self) -> int:
        """Dataset size"""
        return self.rx.size(0)

    def sample(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Sample Data g/ Priorities

        Returns:
            rx (Tensor): ray position query (B, 3)
            rd (Tensor): ray direction query (B, 3)
            delta (Tensor): ray segment lengths (B, )
        """
        idxs = np.random.choice(len(self), size=self.voxels)
        idxs = torch.from_numpy(idxs)

        rx = self.rx[idxs.repeat_interleave(self.samples, dim=0)]
        rx = rx + torch.randn_like(rx) * (.5 * self.voxel_size[None, :])
        
        rd = torch.rand((self.batch_size, 3), dtype=torch.float32)
        rd = rd / torch.linalg.norm(rd, dim=-1, keepdim=True)

        delta = torch.rand((self.batch_size, ), dtype=torch.float32)
        delta = self.domain.d[0] + abs(self.domain.d[1] - self.domain.d[0]) * delta

        return rx, rd, delta


def distill(
    teacher: NeRF,
    student: NeRF,
    sampler: VoxelSampler,
    optim: Optimizer,
    scheduler: Scheduler,
    scaler: GradScaler,
    iterations: Optional[int] = 100,
    callbacks: Optional[List[Callable[[int, History], None]]] = [],
    verbose: bool = True,
) -> History:
    """Distill teacher NeRF into a student

    Arguments:
        teacher (NeRF): neural radiance field teacher model
        student (NeRF): neural radiance field student to be trained
        sampler (VoxelSampler): voxel sampler
        optim (Optimizer): optimization strategy
        scheduler (Scheduler): learning rate scheduler
        scaler (GradScaler): grad scaler for half precision (fp16)
        iterations (Optional[int]): amount of iterations to train (default: 100)
        callbacks (Optional[List[Callable[[int, History], None]]]): callbacks (default: [])
        verbose (Optional[bool]): print tqdm (default: True)

    Returns:
        history (History): training history
    """
    d = next(teacher.parameters()).device
    H = History()

    steps = tqdm(range(iterations), desc="[NeRF] Distill Step", disable=not verbose)
    for step in steps:
        rx, rd, delta = sampler.sample()
        rx, rd, delta = rx.to(d), rd.to(d), delta.to(d)

        with autocast(enabled=scaler.is_enabled()):
            with torch.no_grad():
                t_sigma, t_rgb = teacher(rx, rd)
                t_alpha = 1. - torch.exp(-t_sigma.view(-1, 1) * delta)
                
            s_sigma, s_rgb = student(rx, rd)
            s_alpha = 1. - torch.exp(-s_sigma.view(-1, 1) * delta)

            loss_alpha = torch.mean((s_alpha - t_alpha) ** 2, dim=-1)
            loss_rgb = torch.mean((s_rgb - t_rgb) ** 2, dim=-1)
            loss = torch.mean(loss_alpha + loss_rgb, dim=0)

        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        if GRAD_NORM_CLIP > 0.: clip_grad_norm_(student.parameters(), GRAD_NORM_CLIP)
        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)
        scheduler.step()

        steps.set_postfix(loss=f"{loss.item():.2e}")

        H.train.append((loss.item(), None))
        H.lr.append(scheduler.lr)
        for callback in callbacks:
            callback(step, H)

    return H


NeRF.distill = distill


if __name__ == "__main__":
    import numpy as np
    import os
    import torch.jit as jit

    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    from copy import deepcopy
    from multiprocessing import cpu_count
    from nerf.core.renderer import BoundedVolumeRaymarcher as BVR
    from nerf.core.scheduler import IndendityScheduler, LogDecayScheduler, MipNeRFScheduler
    from nerf.data.blender import BlenderDataset
    from nerf.train import fit
    from nerf.utils.callbacks import plot_distill_callback, plot_train_callback, render_callback
    from torch.nn import MSELoss
    from torch.optim import Adam
    from torch.utils.data import Dataset


    JOBS = cpu_count()
    ITERATIONS = 1_000_000
    LOG = 10_000

    parser = ArgumentParser(__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input",      type=str,   required=True,       help="Blender Dataset Path")
    parser.add_argument("-s", "--scene",      type=str,   required=True,       help="Blender Scene Name")
    parser.add_argument("-o", "--output",     type=str,   required=True,       help="NeRF Student Output Folder")
    parser.add_argument("-t", "--teacher",    type=str,   required=True,       help="NeRF Pretrainted Teacher Path")
    parser.add_argument(      "--iterations", type=int,   default=ITERATIONS,  help="Number of Iteration to train")
    parser.add_argument("-l", "--lr",         type=float, default=5e-4,        help="Starting Learning Rate")
    parser.add_argument(      "--scheduler",  type=str,   default="LogDecay",  help="Learning Rate Scheduler")
    parser.add_argument(      "--amp",                    action="store_true", help="Automatic Mixted Precision")
    parser.add_argument(      "--voxels",     type=int,   default=16,          help="Voxels per Batch")
    parser.add_argument(      "--samples",    type=int,   default=64,          help="Samples per Voxel")
    parser.add_argument(      "--step",       type=int,   default=1,           help="Frame step")
    parser.add_argument(      "--scale",      type=float, default=1.,          help="Frame scale")
    parser.add_argument(      "--near",       type=float, default=2.,          help="Near Plane")
    parser.add_argument(      "--far",        type=float, default=6.,          help="Far Plane")
    parser.add_argument(      "--perturb",                action="store_true", help="Perturb along Ray")
    parser.add_argument("-c", "--coarse",     type=int,   default=64,          help="Coarse samples")
    parser.add_argument("-f", "--fine",       type=int,   default=64,          help="Fine samples")
    parser.add_argument("-e", "--epochs",     type=int,   default=50,          help="Number of Epochs to train")
    parser.add_argument("-b", "--batch_size", type=int,   default=4_096,       help="Batch Size")
    parser.add_argument("-j", "--jobs",       type=int,   default=JOBS,        help="Number of Processes")
    parser.add_argument(      "--log",        type=int,   default=LOG,         help="Log Frequency")
    parser.add_argument("-d", "--device",     type=int,   default=0,           help="Cuda GPU ID (-1 for CPU)")
    args = parser.parse_args()


    def SCHEDULER(optim: Optimizer) -> Scheduler:
        es = .1 * args.iterations
        lr = args.lr * 1e-2, args.lr

        if args.scheduler == "MipNeRF":
            scheduler = MipNeRFScheduler(optim, args.iterations, es, 1., lr_range=lr)
        elif args.scheduler == "LogDecay":
            scheduler = LogDecayScheduler(optim, args.iterations, 1., lr_range=lr)
        else:
            scheduler = IndendityScheduler(optim, args.lr)
        
        return scheduler


    device = "cpu" if args.device < 0 else f"cuda:{args.device}"

    teacher = jit.load(args.teacher).to(device)
    student = NeRF(deepcopy(teacher.phi_x), deepcopy(teacher.phi_d), width=128, depth=2, resid=False).to(device)
    
    optim = Adam(student.parameters(), lr=args.lr, eps=1e-4 if args.amp else 1e-8)
    scheduler = SCHEDULER(optim)
    scaler = GradScaler(enabled=args.amp)

    if not os.path.isdir(args.output):
        os.makedirs(args.output, exist_ok=True)


    APPLY_CBK = lambda s, args: s == 0 or (s + 1) % args.log == 0 or s == args.iterations -1
    
    def SAVE_CBK(step: int, history: History) -> None:
        path = os.path.join(args.output, f"NeRF_{args.scene}.model.ts")
        if APPLY_CBK(step, args):
            jit.save(jit.script(student), path)

    def PLOT_CBK(step: int, history: History) -> None:
        if APPLY_CBK(step, args):
            path = os.path.join(args.output, f"NeRF_{args.scene}.plot.distill.png")
            plot_distill_callback(history, args.scene, path)


    x, d = (-4, 4), ((.6 - .2) / 64, (.6 - .2) / 64)
    domain = Domain(x=x, y=x, z=x, d=d)
    sampler = VoxelSampler(domain, voxels=args.voxels, samples=args.samples)
    
    distill(
        teacher,
        student,
        sampler,
        optim,
        scheduler,
        scaler,
        iterations=args.iterations,
        callbacks=[SAVE_CBK, PLOT_CBK],
    )


    def SCHEDULER(optim: Optimizer, dataset: Dataset) -> Scheduler:
        es = .01 * args.epochs
        spe = len(dataset) // args.batch_size
        spe += 1 * (len(dataset) % args.batch_size > 0)
        lr = args.lr * 1e-2, args.lr

        if args.scheduler == "MipNeRF":
            scheduler = MipNeRFScheduler(optim, args.epochs, es, spe, lr_range=lr)
        elif args.scheduler == "LogDecay":
            scheduler = LogDecayScheduler(optim, args.epochs, spe, lr_range=lr)
        else:
            scheduler = IndendityScheduler(optim, args.lr)

        return scheduler


    trainset = BlenderDataset(args.input, args.scene, split="train", step=args.step, scale=args.scale)
    valset   = BlenderDataset(args.input, args.scene, split="val",   step=args.step, scale=args.scale)
    testset  = BlenderDataset(args.input, args.scene, split="test",  step=args.step, scale=args.scale)

    raymarcher = BVR(args.near, args.far, args.coarse, args.fine)
    criterion = MSELoss(reduction="mean").to(device)

    optim = Adam(student.parameters(), lr=args.lr, eps=1e-4 if args.amp else 1e-8)
    scheduler = SCHEDULER(optim, trainset)
    scaler = GradScaler(enabled=args.amp)

    args.log = int(max(args.log * args.epcohs / args.steps, 1.))
    APPLY_CBK = lambda e, args: e == 0 or (e + 1) % args.log == 0 or e == args.epochs -1

    def RENDER_CBK(epoch: int, history: History) -> None:
        if APPLY_CBK(epoch, args):
            data = valset.ro, valset.rd, valset.C
            size = valset.H, valset.W, args.batch_size
            path = os.path.join(args.output, f"NeRF_{args.scene}.img.png")
            render_callback(student, raymarcher, *data, *size, path)

    def PLOT_CBK(epoch: int, history: History) -> None:
        if APPLY_CBK(epoch, args):
            path = os.path.join(args.output, f"NeRF_{args.scene}.plot.png")
            plot_train_callback(history, args.scene, path)


    fit(
        student,
        raymarcher,
        optim,
        scheduler,
        criterion,
        scaler,
        trainset,
        valset,
        testset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        jobs=args.jobs,
        perturb=args.perturb,
        callbacks=[SAVE_CBK, RENDER_CBK, PLOT_CBK],
    )