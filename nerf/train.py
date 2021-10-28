import numpy as np
import torch
from nerf.core.features import FeatureMapping

from nerf.core.model import NeRF
from nerf.core.renderer import BoundedVolumeRaymarcher as BVR
from nerf.core.scheduler import Scheduler
from nerf.data.utils import loaders
from nerf.utils.history import History
from nerf.utils.pbar import tqdm
from torch import device
from torch.cuda.amp import autocast, GradScaler
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from typing import Callable, List, Optional, Tuple


GRAD_NORM_CLIP = 0.


def step(
    epoch: int,
    nerf: NeRF,
    raymarcher: BVR,
    optim: Optimizer,
    scheduler: Scheduler,
    criterion: Module,
    scaler: GradScaler,
    loader: DataLoader,
    d: device,
    split: str,
    perturb: Optional[bool] = False,
    verbose: Optional[bool] = True,
) -> Tuple[List[float], List[float], List[float]]:
    """Training/Validation/Testing step

    Arguments:
        epoch (int): current epoch
        nerf (NeRF): neural radiance field model to be trained
        raymarcher (BVR): bounded volume raymarching renderer
        optim (Optimizer): optimization strategy
        scheduler (Scheduler): learning rate scheduler
        criterion (Module): objective function
        scaler (GradScaler): grad scaler for half precision (fp16)
        loader (DataLoader): batch data loader
        d (device): torch device to send the batch on
        split (str): step state ("train", "val", "test")
        perturb (Optional[bool]): peturb ray query segment (default: False)
        verbose (Optional[bool]): print tqdm (default: True)

    Returns:
        steps_loss (List[float]): loss
        steps_psnr (List[float]): psnr
        steps_lr (List[float]): learning rate
    """
    train = split == "train"
    nerf = nerf.train(train)
    
    total_loss, total_psnr, total_lr = 0., 0., 0.
    steps_loss, steps_psnr, steps_lr = [], [], []

    desc = f"[NeRF] {split.capitalize()} {epoch + 1}"
    batches = tqdm(loader, desc=desc, disable=not verbose)

    with torch.set_grad_enabled(train):
        for C, ro, rd in batches:
            C, ro, rd = C.to(d), ro.to(d), rd.to(d)

            with autocast(enabled=scaler.is_enabled()):
                *_, C_ = raymarcher.render_volume(nerf, ro, rd, perturb=perturb, train=train)
                loss = criterion(C_, C)
            
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                if GRAD_NORM_CLIP > 0.: clip_grad_norm_(nerf.parameters(), GRAD_NORM_CLIP)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                scheduler.step()
                
                total_lr += scheduler.lr / len(loader)
                steps_lr.append(scheduler.lr)

            with torch.no_grad():
                psnr = -10. * torch.log10(loss)
            
            total_loss += loss.item() / len(loader)
            total_psnr += psnr.item() / len(loader)
            steps_loss.append(loss.item())
            steps_psnr.append(psnr.item())

            batches.set_postfix(loss=total_loss, psnr=total_psnr, lr=total_lr)

    return steps_loss, steps_psnr, steps_lr


def fit(
    nerf: NeRF,
    raymarcher: BVR,
    optim: Optimizer,
    scheduler: Scheduler,
    criterion: Module,
    scaler: GradScaler,
    train_data: Dataset,
    val_data: Optional[Dataset] = None,
    test_data: Optional[Dataset] = None,
    epochs: Optional[int] = 100,
    batch_size: Optional[int] = 32,
    jobs: Optional[int] = 8,
    perturb: Optional[bool] = False,
    callbacks: Optional[List[Callable[[int, History], None]]] = [],
    verbose: bool = True,
) -> History:
    """Fit NeRF on a specific dataset

    Arguments:
        nerf (NeRF): neural radiance field model to be trained
        raymarcher (BVR): bounded volume raymarching renderer
        optim (Optimizer): optimization strategy
        scheduler (Scheduler): learning rate scheduler
        criterion (Module): objective function
        scaler (GradScaler): grad scaler for half precision (fp16)
        train_data (Dataset): training dataset
        val_data (Optional[Dataset]): validation dataset (default: None)
        test_data (Optional[Dataset]): testing dataset (default: None)
        epochs (Optional[int]): amount of epochs to train (default: 100)
        batch_size (Optional[int]): batch size (default: 32)
        jobs (Optional[int]): number of processes to use  (default: 8)
        perturb (Optional[bool]): peturb ray query segment (default: False)
        callbacks (Optional[List[Callable[[int, History], None]]]): callbacks (default: [])
        verbose (Optional[bool]): print tqdm (default: True)

    Returns:
        history (History): training history
    """
    datasets = train_data, val_data, test_data
    train, val, test = loaders(*datasets, batch_size, jobs)

    H = History()
    d = next(nerf.parameters()).device
    args = nerf, raymarcher, optim, scheduler, criterion, scaler

    train_opt = { "split": "train", "perturb": perturb, "verbose": verbose }
    val_opt = { "split": "val", "verbose": verbose }
    test_opt = { "split": "test", "verbose": verbose }

    pbar = tqdm(range(epochs), desc="[NeRF] Epoch", disable=not verbose)
    for epoch in pbar:
        mse, psnr, lr = step(epoch, *args, train, d, **train_opt)
        H.train.append((np.average(mse), np.average(psnr)))
        H.lr += lr
        pbar.set_postfix(mse=np.average(mse), psnr=np.average(psnr))
        
        if val:
            mse, psnr, _ = step(epoch, *args, val, d, **val_opt)
            H.val.append((np.average(mse), np.average(psnr)))
            pbar.set_postfix(mse=np.average(mse), psnr=np.average(psnr))
        
        for callback in callbacks:
            callback(epoch, H)
    
    if test:
        mse, psnr, _ = step(epoch, *args, test, d, **test_opt)
        H.test = np.average(mse), np.average(psnr)
        pbar.set_postfix(mse=np.average(mse), psnr=np.average(psnr))

        for callback in callbacks:
            callback(epoch, H)

    return H


NeRF.fit = fit


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    import torch.jit as jit

    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    from multiprocessing import cpu_count
    from nerf.core.features import FourierFeatures as FF, PositionalEncoding as PE
    from nerf.core.scheduler import IndendityScheduler, LogDecayScheduler, MipNeRFScheduler
    from nerf.data.blender import BlenderDataset
    from nerf.infer import infer
    from nerf.reptile import reptile
    from PIL import Image
    from torch.nn import MSELoss, LeakyReLU, ReLU, SiLU
    from torch.optim import Adam


    JOBS = cpu_count()
    ACTIVATIONS = {a.__name__: a for a in [ReLU, LeakyReLU, SiLU]}


    def SCHEDULER(optim: Optimizer, dataset: Dataset, args) -> Scheduler:
        epochs_shift = .01 * args.epochs
        steps_per_epoch = len(dataset) // args.batch_size
        steps_per_epoch += 1 * (len(dataset) % args.batch_size > 0)
        lr_range = args.lr * 1e-2, args.lr

        if args.scheduler == "MipNeRF":
            return MipNeRFScheduler(
                optim,
                args.epochs,
                epochs_shift,
                steps_per_epoch,
                lr_range=lr_range,
            )

        if args.scheduler == "LogDecay":
            return LogDecayScheduler(
                optim,
                args.epochs,
                steps_per_epoch,
                lr_range=lr_range,
            )

        return IndendityScheduler(optim, args.lr)


    def MAPPING(input_dim: int, args) -> Tuple[FeatureMapping, FeatureMapping]:
        if args.mapping == "FF":
            return (
                FF(input_dim, args.features_x, args.sigma_x),
                FF(input_dim, args.features_d, args.sigma_d),
            )

        return PE(input_dim, args.freqs_x), PE(3, args.freqs_d)


    APPLY_CBK = lambda epoch, args: (
        epoch == 0 or
        (epoch + 1) % args.log == 0 or
        epoch == args.epochs -1
    )


    parser = ArgumentParser(__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input",         type=str,   required=True,       help="Blender Dataset Path")
    parser.add_argument("-s", "--scene",         type=str,   required=True,       help="Blender Scene Name")
    parser.add_argument("-o", "--output",        type=str,   required=True,       help="Model Output Folder")
    parser.add_argument(      "--step",          type=int,   default=1,           help="Frame step")
    parser.add_argument(      "--scale",         type=float, default=1.,          help="Frame scale")
    parser.add_argument(      "--near",          type=float, default=2.,          help="Near Plane")
    parser.add_argument(      "--far",           type=float, default=6.,          help="Far Plane")
    parser.add_argument(      "--perturb",                   action="store_true", help="Perturb along Ray")
    parser.add_argument("-c", "--coarse",        type=int,   default=64,          help="Coarse samples")
    parser.add_argument("-f", "--fine",          type=int,   default=64,          help="Fine samples")
    parser.add_argument(      "--mapping",       type=str,   default="PE",        help="Feature Mapping Method (PE | FF)")
    parser.add_argument(      "--freqs_x",       type=int,   default=10,          help="Positional Encoding frequency band range for query position")
    parser.add_argument(      "--freqs_d",       type=int,   default=4,           help="Positional Encoding frequency band range for query diretion")
    parser.add_argument(      "--features_x",    type=int,   default=256,         help="Fourrier Features features for query position")
    parser.add_argument(      "--features_d",    type=int,   default=64,          help="Fourrier Features features for query diretion")
    parser.add_argument(      "--sigma_x",       type=float, default=32.,         help="Fourrier Features sigma for query position")
    parser.add_argument(      "--sigma_d",       type=float, default=16.,         help="Fourrier Features sigma for query diretion")
    parser.add_argument(      "--nerf_width",    type=int,   default=256,         help="Model Width")
    parser.add_argument(      "--nerf_depth",    type=int,   default=8,           help="Model Depth")
    parser.add_argument(      "--nerf_resid",                action="store_true", help="Model enable Residual")
    parser.add_argument(      "--nerf_activ",    type=str,   default="SiLU",      help="Model Activation Function (LeakyReLU | ReLU | SiLU)")
    parser.add_argument("-e", "--epochs",        type=int,   default=100,         help="Number of Epochs to train")
    parser.add_argument("-l", "--lr",            type=float, default=5e-4,        help="Starting Learning Rate")
    parser.add_argument(      "--scheduler",     type=str,   default="LogDecay",  help="Learning Rate Scheduler")
    parser.add_argument("-r", "--reptile",                   action="store_true", help="Reptile Initialization")
    parser.add_argument(      "--reptile_steps", type=int,   default=16,          help="Reptile Initialization Steps")
    parser.add_argument(      "--amp",                       action="store_true", help="Automatic Mixted Precision")
    parser.add_argument("-b", "--batch_size",    type=int,   default=4_096,       help="Frame width")
    parser.add_argument("-j", "--jobs",          type=int,   default=JOBS,        help="Number of Processes")
    parser.add_argument(      "--log",           type=int,   default=1,           help="Log Frequency")
    parser.add_argument("-d", "--device",        type=int,   default=0,           help="Cuda GPU ID (-1 for CPU)")
    args = parser.parse_args()

    trainset = BlenderDataset(args.input, args.scene, split="train", step=args.step, scale=args.scale)
    valset   = BlenderDataset(args.input, args.scene, split="val",   step=args.step, scale=args.scale)
    testset  = BlenderDataset(args.input, args.scene, split="test",  step=args.step, scale=args.scale)

    device = "cpu" if args.device < 0 else f"cuda:{args.device}"

    nerf = NeRF(
        *MAPPING(3, args),
        width=args.nerf_width,
        depth=args.nerf_depth,
        activ=ACTIVATIONS.get(args.nerf_activ, ReLU),
        resid=args.nerf_resid,
    ).to(device)

    raymarcher = BVR(args.near, args.far, args.coarse, args.fine)
    criterion = MSELoss(reduction="mean").to(device)

    optim = Adam(nerf.parameters(), lr=args.lr, eps=1e-4 if args.amp else 1e-8)
    scheduler = SCHEDULER(optim, trainset, args)
    scaler = GradScaler(enabled=args.amp)

    if not os.path.isdir(args.output):
        os.makedirs(args.output, exist_ok=True)


    MODEL_PATH = os.path.join(args.output, f"NeRF_{args.scene}.model.ts")
    def SAVE_CBK(epoch: int, history: History) -> None:
        if APPLY_CBK(epoch, args): jit.save(jit.script(nerf), MODEL_PATH)


    IMG_PATH = os.path.join(args.output, f"NeRF_{args.scene}.img.png")
    H, W, C = valset.H, valset.W, 3
    def RENDER_CBK(epoch: int, history: History) -> None:
        if APPLY_CBK(epoch, args):
            ro, rd = valset.ro[:H * W], valset.rd[:H * W]
            depth, rgb = infer(nerf, raymarcher, ro, rd, H, W, batch_size=args.batch_size)
            
            depth, rgb = depth.numpy().astype(np.uint8), rgb.numpy().astype(np.uint8)
            gt = (valset.C[:H * W].view(H, W, C) * 255).numpy().astype(np.uint8)
            img = np.hstack((gt, rgb, depth))

            Image.fromarray(img).save(IMG_PATH)


    PLOT_PATH = os.path.join(args.output, f"NeRF_{args.scene}.plot.png")
    def PLOT_CBK(epoch: int, history: History) -> None:
        if APPLY_CBK(epoch, args):
            fig = plt.figure(figsize=(24, 4))

            ax_mse = fig.add_subplot(131)
            ax_psnr = fig.add_subplot(132)
            ax_lr = fig.add_subplot(133)

            ax_mse.set_title(f"NeRF {args.scene.capitalize()} MSE")
            ax_psnr.set_title(f"NeRF {args.scene.capitalize()} PSNR")
            ax_lr.set_title(f"NeRF {args.scene.capitalize()} Learning Rate")
            
            ax_mse.set_ylabel(f"MSE")
            ax_psnr.set_ylabel(f"PSNR")
            ax_lr.set_ylabel(f"LR")

            ax_mse.set_xlabel("epochs")
            ax_psnr.set_xlabel("epochs")
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

            ax_lr.set_xscale("log")
            ax_lr.set_yscale("log")

            ax_mse.grid(linestyle="dotted")
            ax_psnr.grid(linestyle="dotted")
            ax_lr.grid(linestyle="dotted")
            
            ax_mse.legend()
            ax_psnr.legend()

            fig.tight_layout()
            fig.canvas.draw()
            fig.savefig(PLOT_PATH)
        

    if args.reptile:
        H = reptile(
            nerf,
            raymarcher,
            optim,
            criterion,
            scaler,
            trainset,
            epochs=1,
            steps=args.reptile_steps,
            batch_size=args.batch_size,
            jobs=args.jobs,
            perturb=args.perturb,
            callbacks=[SAVE_CBK, RENDER_CBK],
        )

    H = fit(
        nerf,
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