import gc
import matplotlib.pyplot as plt
from nerf.core import scheduler
import nerf.infer
import nerf.reptile
import nerf.train
import matplotlib
import numpy as np
import os
import torch
import torch.jit as jit

from IPython.display import display
from ipywidgets.widgets import Button, Dropdown, Image, Layout, VBox
from nerf.data import BlenderDataset
from nerf.core import NeRF, BoundedVolumeRaymarcher as BVR
from nerf.core.features import PositionalEncoding as PE, FourierFeatures as FF
from nerf.core.scheduler import IndendityScheduler, LogDecayScheduler, MipNeRFScheduler, Scheduler
from nerf.notebook.core.standard import StandardTabsWidget
from nerf.notebook.config.train import TrainConfig
from nerf.train import History
from PIL import Image as PImage
from torch.cuda.amp import GradScaler
from torch.nn import LeakyReLU, MSELoss, ReLU, SiLU
from torch.optim import Adam
from torchsummary import summary


matplotlib.use('Agg')

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)


class Trainer(StandardTabsWidget):
    """Trainer Widget
    
    Arguments:
        config (TrainConfig): Training configuration
        verbose (bool): Verbose for tqdm (default: True)
    """

    def __init__(self, config: TrainConfig, verbose: bool = True) -> None:
        self.loads = sorted(os.listdir(config.root))
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
        self.scheduler: Scheduler = None
        self.scaler: GradScaler = None
        self.history: History = None
        self.callbacks = [self.save_callback, self.render_callback, self.plot_callback]

    def setup_widgets(self) -> None:
        self.register_widget("load", Dropdown(options=self.loads, value=self.loads[0], description="Exp"))
        self.register_widget("btn_load", Button(description="Load", icon="spinner"))

        self.register_widget("btn_dataset", Button(description="Dataset", icon="database", layout=Layout(width="80%", height="100%")))
        self.register_widget("btn_model", Button(description="Model", icon="tasks", layout=Layout(width="80%", height="100%")))
        self.register_widget("btn_raymarcher", Button(description="Raymarcher", icon="cloud", layout=Layout(width="80%", height="100%")))
        self.register_widget("btn_optimsuite", Button(description="Optimizer", icon="spinner", layout=Layout(width="80%", height="100%")))
        self.register_widget("btn_fit", Button(description="Fit", icon="space-shuttle", layout=Layout(width="80%", height="100%")))

        self.register_widget("img_gt", Image(value=b"", format="png", layout=Layout(width="80%")))
        self.register_widget("img_rgb", Image(value=b"", format="png", layout=Layout(width="80%")))
        self.register_widget("img_depth", Image(value=b"", format="png", layout=Layout(width="80%")))
        self.register_widget("img_mse", Image(value=b"", format="png", layout=Layout(width="80%")))
        self.register_widget("img_psnr", Image(value=b"", format="png", layout=Layout(width="80%")))
        self.register_widget("img_lr", Image(value=b"", format="png", layout=Layout(width="80%")))

        self.w_btn_load.on_click(self.setup_load)
        self.w_btn_dataset.on_click(self.setup_dataset)
        self.w_btn_model.on_click(self.setup_model)
        self.w_btn_raymarcher.on_click(self.setup_raymarcher)
        self.w_btn_optimsuite.on_click(self.setup_optimsuite)
        self.w_btn_fit.on_click(self.fit)

        self.w_btn_load.disabled = False
        self.w_btn_dataset.disabled = False
        self.w_btn_model.disabled = True
        self.w_btn_raymarcher.disabled = True
        self.w_btn_optimsuite.disabled = True
        self.w_btn_fit.disabled = True

    def setup_tabs(self) -> None:
        self.register_tab("import", 1, 2, ["load", "btn_load"])
        self.register_tab("actions", 1, 5, ["btn_dataset", "btn_model", "btn_raymarcher", "btn_optimsuite", "btn_fit"])
        self.register_tab("images", 2, 2, ["img_gt", "img_rgb", None, "img_depth"])
        self.register_tab("graphs", 2, 2, ["img_mse", "img_psnr", "img_lr", None])

    def setup_load(self, change) -> None:
        root = os.path.join(self.config.root, self.load())
        cfg = [f for f in os.listdir(root) if f.endswith(".yml")]
        self.config.load(os.path.join(root, cfg[0]))
        print(f"[Config] Loaded Config w/ UUID({self.config.uuid})")
        
        self.config.disable()
        self.w_btn_load.disabled = True
        self.w_btn_dataset.disabled = True
        self.w_btn_model.disabled = True
        self.w_btn_raymarcher.disabled = True
        self.w_btn_optimsuite.disabled = True
        self.w_btn_fit.disabled = True

    def setup_dataset(self, change) -> None:
        if hasattr(self, "trainset"): self.trainset = None
        if hasattr(self, "valset"): self.valset = None
        if hasattr(self, "testset"): self.testset = None

        if not os.path.isdir(self.config.res):
            os.makedirs(self.config.res, exist_ok=True)

        blender = self.config.blender
        scene = self.config.scene()
        step = self.config.step()
        scale = self.config.scale()
        path = self.config.gt_png

        args = blender, scene
        self.trainset = BlenderDataset(*args, "train", step=step, scale=scale)
        self.valset = BlenderDataset(*args, "val", step=step, scale=scale)
        self.testset = BlenderDataset(*args, "test", step=step, scale=scale)

        H, W = self.valset.H, self.valset.W,
        C = self.valset.C[:H * W]

        img = C.view(H, W, 3) * 255
        img = img.numpy().astype(np.uint8)
        img = PImage.fromarray(img)
        img.save(path)

        with open(path, "rb") as f:
            self.w_img_gt.value = f.read()

        print("[Setup] Dataset Ready")

        self.w_btn_load.disabled = True
        self.w_btn_dataset.disabled = True
        self.w_btn_model.disabled = False
        self.w_btn_raymarcher.disabled = True
        self.w_btn_optimsuite.disabled = True
        self.w_btn_fit.disabled = True

    def setup_model(self, change) -> None:
        if hasattr(self, "nerf"):
            self.nerf = None

        if self.config.embedder() == "FourrierFeatures":
            features_x = self.config.features_x()
            features_d = self.config.features_d()
            sigma_x = self.config.sigma_x()
            sigma_d = self.config.sigma_d()

            phi_x = FF(3, features_x, sigma_x)
            phi_d = FF(3, features_d, sigma_d)
        else:
            freqs_x = self.config.freqs_x()
            freqs_d = self.config.freqs_d()

            phi_x = PE(3, freqs_x)
            phi_d = PE(3, freqs_d)
        
        n2a = {a.__name__: a for a in [ReLU, LeakyReLU, SiLU]}

        width = self.config.width()
        depth = self.config.depth()
        activ = n2a.get(self.config.activation(), ReLU)

        self.nerf = NeRF(
            phi_x,
            phi_d,
            width=width,
            depth=depth,
            activ=activ,
        ).cuda()

        print("[Setup] Model Ready")
        summary(self.nerf, [(1, 3), (1, 3)])

        self.w_btn_load.disabled = True
        self.w_btn_dataset.disabled = True
        self.w_btn_model.disabled = True
        self.w_btn_raymarcher.disabled = False
        self.w_btn_optimsuite.disabled = True
        self.w_btn_fit.disabled = True
        
    def setup_raymarcher(self, change) -> None:
        if hasattr(self, "raymarcher"):
            self.raymarcher = None

        tn, tf = self.config.t()
        samples_c = self.config.samples_c()
        samples_f = self.config.samples_f()

        self.raymarcher = BVR(tn, tf, samples_c=samples_c, samples_f=samples_f)
        
        print("[Setup] Raymarcher Ready")

        self.w_btn_load.disabled = True
        self.w_btn_dataset.disabled = True
        self.w_btn_model.disabled = True
        self.w_btn_raymarcher.disabled = True
        self.w_btn_optimsuite.disabled = False
        self.w_btn_fit.disabled = True

    def setup_optimsuite(self, change) -> None:
        if hasattr(self, "criterion"):
            self.criterion = None

        fp16 = self.config.fp16()
        lr = self.config.lr()
        epochs = self.config.epochs()
        eps = 1e-4 if fp16 else 1e-8

        self.criterion = MSELoss(reduction="mean").cuda()
        self.optim = Adam(self.nerf.parameters(), lr=lr, eps=eps)
        
        scheduler = self.config.scheduler()
        if scheduler == "MipNeRF":
            epochs_shift = .01 * epochs
            epochs_shift = epochs_shift if epochs_shift > 0 else 1
            steps_per_epoch = len(self.trainset) // self.config.batch_size()
            steps_per_epoch += 1 * (len(self.trainset) % self.config.batch_size() > 0)
            lr_range = lr * 1e-2, lr

            self.scheduler = MipNeRFScheduler(
                self.optim,
                epochs,
                epochs_shift,
                steps_per_epoch,
                lr_range,
            )

        elif scheduler == "LogDecay":
            steps_per_epoch = len(self.trainset) // self.config.batch_size()
            steps_per_epoch += 1 * (len(self.trainset) % self.config.batch_size() > 0)
            lr_range = lr * 1e-2, lr

            self.scheduler = LogDecayScheduler(
                self.optim,
                epochs,
                steps_per_epoch,
                lr_range,
            )

        else: self.scheduler = IndendityScheduler(self.optim, lr)
        
        self.scaler = GradScaler(enabled=fp16)

        print("[Setup] Optimizer Ready")

        self.w_btn_load.disabled = True
        self.w_btn_dataset.disabled = True
        self.w_btn_model.disabled = True
        self.w_btn_raymarcher.disabled = True
        self.w_btn_optimsuite.disabled = True
        self.w_btn_fit.disabled = False

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
            H, W = self.valset.H, self.valset.W
            ro = self.valset.ro[:H * W]
            rd = self.valset.rd[:H * W]

            batch_size = self.config.batch_size()

            args = self.raymarcher, ro, rd, H, W
            depth_map, rgb_map = self.nerf.infer(*args, batch_size=batch_size, verbose=self.verbose)
            depth_map = depth_map.numpy().astype(np.uint8)
            rgb_map = rgb_map.numpy().astype(np.uint8)
            
            path = self.config.rgb_png
            img = PImage.fromarray(rgb_map)
            img.save(path)

            with open(path, "rb") as f:
                self.w_img_rgb.value = f.read()

            path = self.config.depth_png
            img = PImage.fromarray(depth_map)
            img.save(path)

            with open(path, "rb") as f:
                self.w_img_depth.value = f.read()

            if not self.verbose:
                mse, psnr = history.train[-1]
                print(f"[NeRF] EPOCH:{epoch + 1} - MSE: {mse:.2e} - PSNR: {psnr:.2f}")

    def plot_callback(self, epoch: int, history: History) -> None:
        def plot(name: str, idx: int, widget: Image) -> None:
            title = f"NeRF {self.config.scene().capitalize()} {name.upper()}"
            path = os.path.join(self.config.res, f"NeRF_{self.config.scene()}_{name}.png")
            
            plt.figure()
            plt.title(title)
            plt.ylabel(f"{name.upper()}")
            plt.xlabel("epochs")
            if len(history.train):
                x = np.arange(1, len(history.train) + 1)
                y = np.array([datum[idx] for datum in history.train])
                plt.plot(x, y, label="train")
                plt.xlim((x.min(), x.max()))
            if len(history.val):
                x = np.arange(1, len(history.val) + 1)
                y = np.array([datum[idx] for datum in history.val])
                plt.plot(x, y, label="val")
                plt.xlim((x.min(), x.max()))
            if history.test:
                x = np.arange(1, len(history.train) + 1)
                y = np.array([history.test[idx]] * len(history.train))
                plt.plot(x, y, label="test")
                plt.xlim((x.min(), x.max()))
            plt.grid(linestyle="dotted")
            plt.legend()
            plt.savefig(path)

            with open(path, "rb") as f:
                widget.value = f.read()
        
        plot( "mse", 0, self.w_img_mse)
        plot("psnr", 1, self.w_img_psnr)

        title = f"NeRF {self.config.scene().capitalize()} Learning Rate"
        path = os.path.join(self.config.res, f"NeRF_{self.config.scene()}_lr.png")

        plt.figure()
        plt.title(title)
        plt.ylabel("lr")
        plt.xlabel("steps")
        if len(history.lr):
            x = np.arange(1, len(history.lr) + 1)
            y = np.array(history.lr)
            plt.plot(x, y)
            plt.xlim((x.min(), x.max()))
        plt.xscale("log")
        plt.yscale("log")
        plt.grid(linestyle="dotted")
        plt.savefig(path)

        with open(path, "rb") as f:
            self.w_img_lr.value = f.read()
    
    def fit(self, change) -> None:
        if hasattr(self, "history"):
            self.history = None

        if not os.path.isdir(self.config.res):
            os.makedirs(self.config.res, exist_ok=True)

        self.config.disable()
        self.disable()
        
        self.config.save(self.config.config_yml)

        strategy = self.config.strategy()
        epochs = self.config.epochs()
        steps = self.config.steps()
        batch_size = self.config.batch_size()
        jobs = self.config.jobs()
        perturb = self.config.perturb()
        
        print(f"[Init] {strategy}")
        if strategy == "Reptile":
            self.history = self.nerf.reptile(
                self.raymarcher,
                self.optim,
                self.criterion,
                self.scaler,
                self.trainset,
                epochs=1,
                steps=steps,
                batch_size=batch_size,
                jobs=jobs,
                perturb=perturb,
                callbacks=self.callbacks,
                verbose=self.verbose,
            )

        print(f"[Train] Fitting")
        self.history = self.nerf.fit(
            self.raymarcher,
            self.optim,
            self.scheduler,
            self.criterion,
            self.scaler,
            self.trainset,
            self.valset,
            self.testset,
            epochs=epochs,
            batch_size=batch_size,
            jobs=jobs,
            perturb=perturb,
            callbacks=self.callbacks,
            verbose=self.verbose,
        )
        print(f"[Train] Done")

        self.config.enable()
        self.enable()

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
