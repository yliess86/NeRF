import gc
import matplotlib.pyplot as plt
import nerf.infer
import nerf.train
import numpy as np
import os
import torch
import torch.jit as jit

from IPython.display import display
from ipywidgets.widgets import Button, Image, Layout, VBox
from nerf.data import BlenderDataset
from nerf.core import NeRF, BoundedVolumeRaymarcher as BVR
from nerf.notebook.core.standard import StandardTabsWidget
from nerf.notebook.config.train import TrainConfig
from nerf.train import History
from PIL import Image as PImage
from torch.cuda.amp import GradScaler
from torch.nn import MSELoss
from torch.optim import Adam


torch.backends.cudnn.benchmark = True


class Trainer(StandardTabsWidget):
    """Trainer Widget
    
    Arguments:
        config (TrainConfig): Training configuration
        verbose (bool): Verbose for tqdm (default: True)
    """

    def __init__(self, config: TrainConfig, verbose: bool = True) -> None:
        super().__init__()
        self.config = config
        self.verbose = verbose

        self.trainset: BlenderDataset = None
        self.valset: BlenderDataset = None
        self.testset: BlenderDataset = None
        self.nerf: NeRF = None
        self.raymarcher: BVR = None
        self.criterion: MSELoss = None
        self.optim: Adam = None
        self.scaler: GradScaler = None
        self.history: History = None
        self.callbacks = [self.save_callback, self.render_callback, self.plot_callback]

    def setup_widgets(self) -> None:
        self.register_widget("btn_dataset", Button(description="Dataset", icon="database", layout=Layout(width="80%", height="100%")))
        self.register_widget("btn_model", Button(description="Model", icon="tasks", layout=Layout(width="80%", height="100%")))
        self.register_widget("btn_raymarcher", Button(description="Raymarcher", icon="cloud", layout=Layout(width="80%", height="100%")))
        self.register_widget("btn_optimsuite", Button(description="Optimizer", icon="spinner", layout=Layout(width="80%", height="100%")))
        self.register_widget("btn_fit", Button(description="Fit", icon="space-shuttle", layout=Layout(width="80%", height="100%")))

        self.register_widget("img_gt", Image(value=b"", format="png", layout=Layout(width="80%")))
        self.register_widget("img_pred", Image(value=b"", format="png", layout=Layout(width="80%")))
        self.register_widget("img_mse", Image(value=b"", format="png", layout=Layout(width="80%")))
        self.register_widget("img_psnr", Image(value=b"", format="png", layout=Layout(width="80%")))

        self.w_btn_dataset.on_click(self.setup_dataset)
        self.w_btn_model.on_click(self.setup_model)
        self.w_btn_raymarcher.on_click(self.setup_raymarcher)
        self.w_btn_optimsuite.on_click(self.setup_optimsuite)
        self.w_btn_fit.on_click(self.fit)

    def setup_tabs(self) -> None:
        self.register_tab("actions", 1, 5, ["btn_dataset", "btn_model", "btn_raymarcher", "btn_optimsuite", "btn_fit"])
        self.register_tab("viz", 2, 2, ["img_gt", "img_pred", "img_mse", "img_psnr"])

    def setup_dataset(self, change) -> None:
        if hasattr(self, "trainset"): self.trainset = None
        if hasattr(self, "valset"): self.valset = None
        if hasattr(self, "testset"): self.testset = None

        blender = self.config.blender
        scene = self.config.scene()
        step = self.config.step()
        scale = self.config.scale()
        path = self.config.gt_png

        args = blender, scene
        self.trainset = BlenderDataset(*args, "train", step=step, scale=scale)
        self.valset = BlenderDataset(*args, "val", step=step, scale=scale)
        self.testset = BlenderDataset(*args, "test", step=step, scale=scale)

        W, H = self.valset.W, self.valset.H,
        C = self.valset.C[:H * W]

        img = C.view(W, H, 3) * 255
        img = img.numpy().astype(np.uint8)
        img = PImage.fromarray(img)
        img.save(path)

        with open(path, "rb") as f:
            self.w_img_gt.value = f.read()

        print("[Setup] Dataset Ready")

    def setup_model(self, change) -> None:
        if hasattr(self, "nerf"):
            self.nerf = None

        features = (self.config.features(), ) * 2
        sigma = (self.config.sigma(), ) * 2
        width = self.config.width()
        depth = self.config.depth()

        self.nerf = NeRF(*features, *sigma, width=width, depth=depth).cuda()

        print("[Setup] Model Ready")
        
    def setup_raymarcher(self, change) -> None:
        if hasattr(self, "raymarcher"):
            self.raymarcher = None

        tn, tf = self.config.t()
        samples_c = self.config.samples_c()
        samples_f = self.config.samples_f()

        self.raymarcher = BVR(tn, tf, samples_c=samples_c, samples_f=samples_f)
        
        print("[Setup] Raymarcher Ready")

    def setup_optimsuite(self, change) -> None:
        if hasattr(self, "criterion"):
            self.criterion = None

        fp16 = self.config.fp16()
        lr = self.config.lr()
        eps = 1e-4 if fp16 else 1e-8

        self.criterion = MSELoss().cuda()
        self.optim = Adam(self.nerf.parameters(), lr=lr, eps=eps)
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
            W, H = self.valset.W, self.valset.H
            ro = self.valset.ro[:W * H]
            rd = self.valset.rd[:W * H]

            batch_size = self.config.batch_size()
            path = self.config.pred_png

            args = self.raymarcher, ro, rd, W, H
            pred = self.nerf.infer(*args, batch_size=batch_size, verbose=self.verbose)
            pred = pred.numpy().astype(np.uint8)
            
            img = PImage.fromarray(pred)
            img.save(path)

            with open(path, "rb") as f:
                self.w_img_pred.value = f.read()

            if not self.verbose:
                mse, psnr = history.train[-1]
                print(f"[NeRF] EPOCH:{epoch + 1} - MSE: {mse:.2e} - PSNR: {psnr:.2f}")

    def plot_callback(self, epoch: int, history: History) -> None:
        title = f"NeRF {self.config.scene().capitalize()} MSE"
        path = os.path.join(self.config.res, f"NeRF_{self.config.scene()}_mse.png")
        
        plt.figure()
        plt.title(title)
        plt.ylabel("MSE")
        plt.xlabel("epochs")
        if len(history.train): plt.plot([mse for mse, _ in history.train], label="train")
        if len(history.val): plt.plot([mse for mse, _ in history.val], label="val")
        if history.test: plt.plot([history.test[0]] * len(history.train), label="test")
        plt.legend()
        plt.savefig(path)

        with open(path, "rb") as f:
            self.w_img_mse.value = f.read()

        title = f"NeRF {self.config.scene().capitalize()} PSNR"
        path = os.path.join(self.config.res, f"NeRF_{self.config.scene()}_psnr.png")
        
        plt.figure()
        plt.title(title)
        plt.ylabel("PSNR")
        plt.xlabel("epochs")
        if len(history.train): plt.plot([psnr for _, psnr in history.train], label="train")
        if len(history.val): plt.plot([psnr for _, psnr in history.val], label="val")
        if history.test: plt.plot([history.test[1]] * len(history.train), label="test")
        plt.legend()
        plt.savefig(path)

        with open(path, "rb") as f:
            self.w_img_psnr.value = f.read()
    
    def fit(self, change) -> None:
        if hasattr(self, "history"):
            self.history = None

        self.config.disable()
        self.disable()
        
        self.config.save(self.config.config_yml)

        epochs = self.config.epochs()
        batch_size = self.config.batch_size()
        jobs = self.config.jobs()
        perturb = self.config.perturb()
        meta = self.config.meta()
        meta_steps = self.config.meta_steps()

        print("[Train] Fitting")
        self.history = self.nerf.fit(
            self.raymarcher,
            self.optim,
            self.criterion,
            self.scaler,
            self.trainset,
            self.valset,
            self.testset,
            epochs=epochs,
            batch_size=batch_size,
            jobs=jobs,
            perturb=perturb,
            meta=meta,
            meta_steps=meta_steps,
            callbacks=self.callbacks,
            verbose=self.verbose,
        )
        print("[Train] Fitting Done")

    def clean(self) -> None:
        self.trainset = None
        self.valset = None
        self.testset = None
        self.nerf = None
        self.raymarcher = None
        self.criterion = None
        self.optim = None
        self.scaler = None
        self.history = None

        torch.cuda.empty_cache()
        gc.collect()

        self.config.enable()
        self.enable()

    def display(self) -> None:
        display(VBox([self.config.app, self.app]))
