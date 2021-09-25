import nerf.infer
import nerf.train
import numpy as np
import torch
import torch.jit as jit

from IPython.display import display
from ipywidgets.widgets import Button, Image, Layout, VBox
from nerf.data import BlenderDataset
from nerf.core import NeRF, BoundedVolumeRaymarcher as BVR
from nerf.train import History
from nerf_gui.core.standard import StandardTabsWidget
from nerf_gui.config.train import TrainConfig
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

        self.dataset: BlenderDataset = None
        self.nerf: NeRF = None
        self.raymarcher: BVR = None
        self.criterion: MSELoss = None
        self.optim: Adam = None
        self.scaler: GradScaler = None
        self.history: History = None
        self.callbacks = [self.save_callback, self.render_callback]

    def setup_widgets(self) -> None:
        self.register_widget("btn_dataset", Button(description="Dataset", icon="database", layout=Layout(width="80%", height="100%")))
        self.register_widget("btn_model", Button(description="Model", icon="tasks", layout=Layout(width="80%", height="100%")))
        self.register_widget("btn_raymarcher", Button(description="Raymarcher", icon="cloud", layout=Layout(width="80%", height="100%")))
        self.register_widget("btn_optimsuite", Button(description="Optimizer", icon="spinner", layout=Layout(width="80%", height="100%")))
        self.register_widget("btn_fit", Button(description="Fit", icon="space-shuttle", layout=Layout(width="80%", height="100%")))

        self.register_widget("img_gt", Image(value=b"", format="png", width=256, height=256, layout=Layout(width="80%")))
        self.register_widget("img_pred", Image(value=b"", format="png", width=256, height=256, layout=Layout(width="80%")))

        self.w_btn_dataset.on_click(self.setup_dataset)
        self.w_btn_model.on_click(self.setup_model)
        self.w_btn_raymarcher.on_click(self.setup_raymarcher)
        self.w_btn_optimsuite.on_click(self.setup_optimsuite)
        self.w_btn_fit.on_click(self.fit)

    def setup_tabs(self) -> None:
        self.register_tab("actions", 1, 5, ["btn_dataset", "btn_model", "btn_raymarcher", "btn_optimsuite", "btn_fit"])
        self.register_tab("viz", 1, 2, ["img_gt", "img_pred"])

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
            self.w_img_gt.value = f.read()

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
                self.w_img_pred.value = f.read()

            if not self.verbose:
                mse, psnr = history.train[-1]
                print(f"[NeRF] EPOCH:{epoch + 1} - MSE: {mse:.2e} - PSNR: {psnr:.2f}")
    
    def fit(self, change) -> None:
        if hasattr(self, "history"):
            del self.history

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
            self.dataset,
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

    def display(self) -> None:
        display(VBox([self.config.app, self.app]))
