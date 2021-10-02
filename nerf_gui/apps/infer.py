import nerf.infer
import numpy as np
import torch
import torch.jit as jit

from IPython.display import display
from ipywidgets.widgets import Button, FloatSlider, FloatRangeSlider, Image, Layout, VBox
from moviepy.editor import ImageSequenceClip
from nerf.data import BlenderDataset
from nerf.core import NeRF, BoundedVolumeRaymarcher as BVR
from nerf.utils.pbar import tqdm
from nerf_gui.config.infer import InferConfig
from nerf_gui.core.standard import StandardTabsWidget
from PIL import Image as PImage


torch.backends.cudnn.benchmark = True


class Inferer(StandardTabsWidget):
    """Inferer Widget
    
    Arguments:
        config (InferConfig): Inference configuration
        verbose (bool): Verbose for tqdm (default: True)
    """

    def __init__(self, config: InferConfig, verbose: bool = True) -> None:
        super().__init__()
        self.config = config
        self.verbose = verbose

        self.dataset: BlenderDataset = None
        self.nerf: NeRF = None
        self.raymarcher: BVR = None

    def setup_widgets(self) -> None:
        self.register_widget("btn_dataset", Button(description="Dataset", icon="database", layout=Layout(width="80%", height="100%")))
        self.register_widget("btn_model", Button(description="Model", icon="tasks", layout=Layout(width="80%", height="100%")))
        self.register_widget("btn_raymarcher", Button(description="Raymarcher", icon="cloud", layout=Layout(width="80%", height="100%")))
        self.register_widget("btn_render", Button(description="Render", icon="camera", layout=Layout(width="80%", height="100%")))

        self.register_widget("theta", FloatRangeSlider(min=-2. * np.pi, max=2. * np.pi, step=.1, value=[-1 / 6 * np.pi, -1 / 6 * np.pi], description="Theta"))
        self.register_widget("phi", FloatRangeSlider(min=-2. * np.pi, max=2. * np.pi, step=.1, value=[-np.pi, np.pi], description="Phi"))
        self.register_widget("radius", FloatSlider(min=0., max=10., step=.1, value=4., description="Radius"))

        self.register_widget("gif_pred", Image(value=b"", format="gif", width=256, height=256, layout=Layout(width="80%")))
        
        self.w_btn_dataset.on_click(self.setup_dataset)
        self.w_btn_model.on_click(self.setup_model)
        self.w_btn_raymarcher.on_click(self.setup_raymarcher)
        self.w_btn_render.on_click(self.render)

    def setup_tabs(self) -> None:
        self.register_tab("actions", 1, 4, ["btn_dataset", "btn_model", "btn_raymarcher", "btn_render"])
        self.register_tab("traj", 1, 3, ["radius", "theta", "phi"])
        self.register_tab("viz", 1, 1, ["gif_pred"])

    def setup_dataset(self, change) -> None:
        if hasattr(self, "dataset"):
            del self.dataset

        blender = self.config.blender
        scene = self.config.scene()
        scale = self.config.scale()

        args = blender, scene, "train"
        self.dataset = BlenderDataset(*args, step=None, scale=scale)

        print("[Setup] Dataset Ready")

    def setup_model(self, change) -> None:
        if hasattr(self, "nerf"):
            del self.nerf

        path = self.config.model_ts
        self.nerf = jit.load(path).cuda()
        self.nerf.infer = lambda *args, **kwargs: NeRF.infer(self.nerf, *args, **kwargs)
        
        print("[Setup] Model Ready")

    def setup_raymarcher(self, change) -> None:
        if hasattr(self, "raymarcher"):
            del self.raymarcher

        tn, tf = self.config.t()
        samples_c = self.config.samples_c()
        samples_f = self.config.samples_f()

        self.raymarcher = BVR(tn, tf, samples_c=samples_c, samples_f=samples_f)
        
        print("[Setup] Raymarcher Ready")

    def render(self, change) -> None:
        self.config.disable()
        self.disable()

        frames = self.config.frames()
        fps = self.config.fps()
        batch_size = self.config.batch_size()
        path = self.config.pred_gif

        theta = self.w_theta.value
        phi = self.w_phi.value
        radius = self.w_radius.value

        W, H = self.dataset.W, self.dataset.H
        S = W * H

        print("[Setup] Creating Rays")
        args = theta, phi, radius, frames
        ros, rds = self.dataset.turnaround_data(*args)

        print("[Setup] Rendering Started")
        preds = np.zeros((frames, W, H, 3), dtype=np.uint8)
        pbar = tqdm(range(0, len(ros), S), desc="[NeRF] Frame", disable=not self.verbose)
        for i, s in enumerate(pbar):
            ro, rd = ros[s:s + S], rds[s:s + S]

            args = self.raymarcher, ro, rd, W, H
            pred = self.nerf.infer(*args, batch_size=batch_size, verbose=self.verbose)
            preds[i] = pred.numpy().astype(np.uint8)

            PImage.fromarray(preds[i]).save(path)
            with open(path, "rb") as f:
                self.w_gif_pred.value = f.read()    

        print("[Setup] Rendering Done ")
        clip = ImageSequenceClip(list(preds), durations=[1. / fps] * frames)
        clip.write_gif(path, fps=fps, verbose=self.verbose)

        with open(path, "rb") as f:
            self.w_gif_pred.value = f.read()

    def display(self) -> None:
        display(VBox([self.config.app, self.app]))