"""python3 -m nerf.bench

Benchmark NeRF performances in seconds (s) and frames per second (FPS).
+---------+---------+---------+---------+
| Metric  |   min   |   avg   |   max   |
+---------+---------+---------+---------+
| Seconds |         |         |         |
+---------+---------+---------+---------+
| FPS     |         |         |         |
+---------+---------+---------+---------+
"""
import numpy as np
import torch

from nerf.core.model import NeRF
from nerf.core.renderer import BoundedVolumeRaymarcher as BVR
from nerf.utils.pbar import tqdm
from prettytable import PrettyTable
from time import time
from torch.cuda.amp import autocast
from typing import NamedTuple


F16_BYTES, F32_BYTES = 2, 4


class BenchmarkConfig(NamedTuple):
    """Benchmark Configuration

    Arguments:
        H (int): image height
        W (int): image width
        B (int): batch size
    """
    H: int
    W: int
    B: int


class BenchmarkData(NamedTuple):
    """Benchmark Data

    Arguments:
        min (float): minimum value
        avg (float): average value
        max (float): maximum value
    """
    min: float
    avg: float
    max: float


class BenchmarkRecord(NamedTuple):
    """Benchmark Record

    Arguments:
        time (BenchmarkData): time record (in seconds)
        fps (BenchmarkData): frame per second record (in fps)
    """
    time: BenchmarkData
    fps: BenchmarkData


class Benchmark:
    """Benchmark

    Arguments:
        config (BenchmarkConfig): benchmark config
        trials (int): number of trials (default: 100)
    """

    def __init__(self, config: BenchmarkConfig, trials: int = 100) -> None:
        self.config = config
        self.trials = trials

    def process(self) -> None:
        """Process to benchmark"""
        raise NotImplementedError("Process not Implemented yet!")

    def profile(self) -> float:
        """Profile Process
        
        Returns:
            dt (float): time spent processing
        """
        t_start = time()
        self.process()
        t_end = time()
        return t_end - t_start

    def run(self) -> BenchmarkRecord:
        """Run benchmark
        
        Returns:
            record (BenchmarkRecord): collected benchmark record
        """
        name = self.__class__.__name__
        pbar = tqdm(range(self.trials), desc=f"[Bench] {name}")
        profiles = [self.profile() for _ in pbar]
        
        min_time = np.min(profiles)
        avg_time = np.average(profiles)
        max_time = np.max(profiles)
        time = BenchmarkData(min_time, avg_time, max_time)

        min_fps = 1. / max_time
        avg_fps = 1. / avg_time
        max_fps = 1. / min_time
        fps = BenchmarkData(min_fps, avg_fps, max_fps)

        return BenchmarkRecord(time, fps)


class BenchmarkInference(Benchmark):
    """"Inference Benchmark

    Arguments:
        coarse (NeRF): coarse pretrained nerf
        fine (NeRF): fine pretrained nerf
        raymarcher (BVR): scene raymarcher
        config (BenchmarkConfig): benchmark config
        trials (int): number of trials (default: 100)
        res (int): float resolution (default: 4)
    """
    
    def __init__(
        self,
        coarse: NeRF,
        fine: NeRF,
        raymarcher: BVR,
        config: BenchmarkConfig,
        trials: int = 100,
        res: int = F32_BYTES,
    ) -> None:
        super().__init__(config, trials)
        self.coarse = coarse
        self.fine = fine
        self.raymarcher = raymarcher
        self.res = res

        self.device = next(coarse.parameters()).device

        self.ro = torch.rand((self.config.H * self.config.W, 3), device=self.device)
        self.rd = torch.rand((self.config.H * self.config.W, 3), device=self.device)
        
        self.rgb = torch.zeros((self.config.H * self.config.W, 3), device=self.device)
        self.depth = torch.zeros((self.config.H * self.config.W), device=self.device)

    @torch.inference_mode()
    def process(self) -> None:
        """Rendering process to benchmark"""
        B = self.config.B
        n = self.ro.size(0)

        with autocast(enabled=self.res == F16_BYTES):
            for s in range(0, n, B):
                e = min(s + B, n)
                
                ro, rd = self.ro[s:e], self.rd[s:e]
                *_, D, C = self.raymarcher.render_volume(self.coarse, self.fine, ro, rd)
                self.rgb[s:e], self.depth[s:e] = C, D

        self.depth = self.depth.nan_to_num()
        self.depth = self.depth - self.depth.min()
        self.depth = self.depth / (self.depth.max() + 1e-10)
        self.depth = self.depth.clip(0, 1) * 255

        self.rgb = self.rgb.clip(0, 1) * 255


def benchmark(
    coarse: NeRF,
    fine: NeRF,
    raymarcher: BVR,
    H: int,
    W: int,
    batch_size: int,
    trials: int = 100,
    res: int = F32_BYTES,
) -> str:
    """"Benchmark

    Arguments:
        coarse (NeRF): coarse pretrained nerf
        fine (NeRF): fine pretrained nerf
        raymarcher (BVR): scene raymarcher
        config (BenchmarkConfig): benchmark config
        trials (int): number of trials (default: 100)
        res (int): float resolution (default: 4)

    Returns:
        summary (str): benchmark summary
    """
    args = coarse, fine, raymarcher

    config = BenchmarkConfig(H, W, batch_size)
    rec = BenchmarkInference(*args, config, trials, res).run()
    format = lambda x: f"{x:.2f}"

    summary = PrettyTable()
    summary.field_names = ["Metric", "min", "avg", "max"]
    summary.add_row(["Seconds", *map(format, rec.time)])
    summary.add_row(["FPS", *map(format, rec.fps)])
    return summary


def size(nerf: NeRF, res: int = F32_BYTES) -> int:
    """Measure NeRF Size

    Arguments:
        nerf (NeRF): pretrained nerf
        res (int): float resolution (default: 4)

    Returns:
        size (int): size in bytes
    """
    return np.sum([np.product(p.size()) * res for p in nerf.parameters()])


if __name__ == "__main__":
    import torch.jit as jit

    from argparse import ArgumentParser, RawDescriptionHelpFormatter


    parser = ArgumentParser(__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input",      type=str,   required=True,  nargs="+", help="TorchScript - NeRF path | Coarse and Fine NeRF paths ")
    parser.add_argument(      "--height",     type=int,   default=800,               help="Frame height")
    parser.add_argument(      "--width",      type=int,   default=800,               help="Frame width")
    parser.add_argument(      "--near",       type=float, default=2.,                help="Near Plane")
    parser.add_argument(      "--far",        type=float, default=6.,                help="Far Plane")
    parser.add_argument("-c", "--coarse",     type=int,   default=64,                help="Coarse samples")
    parser.add_argument("-f", "--fine",       type=int,   default=64,                help="Fine samples")
    parser.add_argument("-b", "--batch_size", type=int,   default=16_384,            help="Batch Size")
    parser.add_argument("-t", "--trials",     type=int,   default=10,                help="Benchmark trials")
    parser.add_argument("-d", "--device",     type=int,   default=0,                 help="Cuda GPU ID (-1 for CPU)")
    args = parser.parse_args()

    device = "cpu" if args.device < 0 else f"cuda:{args.device}"
    
    print(args.input)
    coarse, fine = args.input[:2] if len(args.input) > 1 else (args.input * 2)[:2]
    coarse, fine = jit.load(coarse).to(device), jit.load(fine).to(device)

    for name, res in [("f32", F32_BYTES), ("f16", F16_BYTES)]:
        print(f"[Bench({name})] Coarse Model Size: {size(coarse, res=res) / 1e6:.2f} MB")
        print(f"[Bench({name})] Fine Model Size: {size(fine, res=res) / 1e6:.2f} MB")
        
        print(f"[Bench({name})] Coarse Only")
        print(benchmark(
            coarse,
            fine,
            BVR(args.near, args.far, args.coarse, 0),
            args.height,
            args.width,
            args.batch_size,
            trials=args.trials,
            res=res,
        ))

        print(f"[Bench({name})] Coarse and Fine")
        print(benchmark(
            coarse,
            fine,
            BVR(args.near, args.far, args.coarse, args.fine),
            args.height,
            args.width,
            args.batch_size,
            trials=args.trials,
            res=res,
        ))