import nerf.infer
import nerf.train
import numpy as np
import torch
import torch.jit as jit

from IPython.display import display
from ipynb.config import TrainConfig
from ipywidgets import GridspecLayout
from ipywidgets.widgets import Button, Image, Tab, Text, VBox
from nerf.data import BlenderDataset
from nerf.core import NeRF, BoundedVolumeRaymarcher as BVR
from nerf.train import History
from PIL import Image as PImage
from torch.cuda.amp import GradScaler
from torch.nn import MSELoss
from torch.optim import Adam


torch.backends.cudnn.benchmark = True


class Trainer:
    def __init__(self, config: TrainConfig, verbose: bool = True) -> None:
        self.config = config
        self.verbose = verbose

        self.dataset: BlenderDataset = None
        self.nerf: NeRF = None
        self.raymarcher: BVR = None
        self.criterion: MSELoss = None
        self.optim: Adam = None
        self.scaler: GradScaler = None
        self.history: History = None
        self.callbacks = [self.save_callback, self.render_callback]
        
        self.setup_widgets()
        self.setup_layouts()

        self.app = Tab(children=[self.setup, self.btn_fit, self.viz])
        self.app.set_title(0, "Setup")
        self.app.set_title(1, "Train")
        self.app.set_title(2, "Viz")

    def setup_widgets(self) -> None:
        self.btn_dataset = Button(description="Setup Dataset")
        self.btn_model = Button(description="Setup Model")
        self.btn_raymarcher = Button(description="Setup Raymarcher")
        self.btn_optimsuite = Button(description="Setup Optimizer")
        self.btn_fit = Button(description="Fit")

        self.btn_dataset.on_click(self.setup_dataset)
        self.btn_model.on_click(self.setup_model)
        self.btn_raymarcher.on_click(self.setup_raymarcher)
        self.btn_optimsuite.on_click(self.setup_optimsuite)
        self.btn_fit.on_click(self.fit)

        self.img_gt = Image(value=b"", format="png", width=256, height=256)
        self.img_pred = Image(value=b"", format="png", width=256, height=256)

    def setup_layouts(self) -> None:
        self.setup = GridspecLayout(1, 4)
        self.setup[0, 0] = self.btn_dataset
        self.setup[0, 1] = self.btn_model
        self.setup[0, 2] = self.btn_raymarcher
        self.setup[0, 3] = self.btn_optimsuite
        
        self.viz = GridspecLayout(1, 2)
        self.viz[0, 0] = self.img_gt
        self.viz[0, 1] = self.img_pred

    def setup_dataset(self, change) -> None:
        if hasattr(self, "dataset"):
            del self.dataset

        blender = self.config.blender
        scene = self.config.scene()
        step = self.config.step()
        scale = self.config.scale()
        path = self.config.gt_png

        args = blender, scene, "train"
        self.dataset = BlenderDataset(*args, step=step, scale=scale)

        W, H = self.dataset.W, self.dataset.H,
        C = self.dataset.C[:H * W]

        img = C.view(W, H, 3) * 255
        img = img.numpy().astype(np.uint8)
        img = PImage.fromarray(img)
        img.save(path)

        with open(path, "rb") as f:
            self.img_gt.value = f.read()

        print("[Setup] Dataset Ready")

    def setup_model(self, change) -> None:
        if hasattr(self, "nerf"):
            del self.nerf

        features = (self.config.features(), ) * 2
        sigma = (self.config.sigma(), ) * 2
        width = self.config.width()
        depth = self.config.depth()

        self.nerf = NeRF(*features, *sigma, width=width, depth=depth).cuda()

        print("[Setup] Model Ready")
        
    def setup_raymarcher(self, change) -> None:
        if hasattr(self, "raymarcher"):
            del self.raymarcher

        tn, tf = self.config.t()
        samples = self.config.samples()

        self.raymarcher = BVR(tn, tf, samples=samples)
        
        print("[Setup] Raymarcher Ready")

    def setup_optimsuite(self, change) -> None:
        if hasattr(self, "criterion"):
            del self.criterion

        lr = self.config.lr()
        fp16 = self.config.fp16()

        self.criterion = MSELoss().cuda()
        self.optim = Adam(self.nerf.parameters(), lr=lr)
        self.scaler = GradScaler(enabled=fp16)

        print("[Setup] Optimizer Ready")

    def do_callback(self, epoch: int) -> bool:
        epochs = self.config.epochs()
        log = self.config.log()

        return (
            epoch == 0 or
            (epoch + 1) % log == 0 or
            epoch == (epochs - 1)
        )

    def save_callback(self, epoch: int, history: History) -> None:
        if self.do_callback(epoch):
            path = self.config.model_ts
            jit.save(jit.script(self.nerf), path)

    def render_callback(self, epoch: int, history: History) -> None:
        if self.do_callback(epoch):
            W, H = self.dataset.W, self.dataset.H
            ro = self.dataset.ro[:W * H]
            rd = self.dataset.rd[:W * H]

            batch_size = self.config.batch_size()
            path = self.config.pred_png

            args = self.raymarcher, ro, rd, W, H
            pred = self.nerf.infer(*args, batch_size=batch_size, verbose=self.verbose)
            pred = pred.numpy().astype(np.uint8)
            
            img = PImage.fromarray(pred)
            img.save(path)

            with open(path, "rb") as f:
                self.img_pred.value = f.read()

            if not self.verbose:
                mse, psnr = history.train[-1]
                print(f"[NeRF] EPOCH:{epoch + 1} - MSE: {mse:.2e} - PSNR: {psnr:.2f}")
    
    def fit(self, change) -> None:
        if hasattr(self, "history"):
            del self.history

        epochs = self.config.epochs()
        batch_size = self.config.batch_size()
        jobs = self.config.jobs()
        perturb = self.config.perturb()

        print("[Train] Fitting")
        self.history = self.nerf.fit(
            self.raymarcher,
            self.optim,
            self.criterion,
            self.scaler,
            self.dataset,
            epochs=epochs,
            batch_size=batch_size,
            jobs=jobs,
            perturb=perturb,
            callbacks=self.callbacks,
            verbose=self.verbose,
        )
        print("[Train] Fitting Done")

    def display(self) -> None:
        display(VBox([self.config.app, self.app]))
