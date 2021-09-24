import nerf.infer
import numpy as np
import torch
import torch.jit as jit

from IPython.display import display
from ipynb.config import InferConfig
from ipywidgets import GridspecLayout
from ipywidgets.widgets import Button, FloatSlider, FloatRangeSlider, Image, Tab, Text, VBox
from moviepy.editor import ImageSequenceClip
from nerf.data import BlenderDataset
from nerf.core import NeRF, BoundedVolumeRaymarcher as BVR
from nerf.utils.pbar import tqdm
from PIL import Image as PImage 


torch.backends.cudnn.benchmark = True


class Inferer:
    def __init__(self, config: InferConfig, verbose: bool = True) -> None:
        self.config = config
        self.verbose = verbose

        self.dataset: BlenderDataset = None
        self.nerf: NeRF = None
        self.raymarcher: BVR = None

        self.setup_widgets()
        self.setup_layouts()

        self.app = Tab(children=[self.setup, self.traj, self.btn_render, self.viz])
        self.app.set_title(0, "Setup")
        self.app.set_title(1, "Trajectory")
        self.app.set_title(2, "Render")
        self.app.set_title(3, "Viz")

    def setup_widgets(self) -> None:
        self.btn_dataset = Button(description="Setup Dataset")
        self.btn_model = Button(description="Setup Model")
        self.btn_raymarcher = Button(description="Setup Raymarcher")
        self.btn_render = Button(description="Render")

        self.btn_dataset.on_click(self.setup_dataset)
        self.btn_model.on_click(self.setup_model)
        self.btn_raymarcher.on_click(self.setup_raymarcher)
        self.btn_render.on_click(self.render)

        self.w_theta = FloatRangeSlider(min=-2. * np.pi, max=2. * np.pi, step=.1, value=[-1 / 6 * np.pi, -1 / 6 * np.pi], description="Theta")
        self.w_phi = FloatRangeSlider(min=-2. * np.pi, max=2. * np.pi, step=.1, value=[-np.pi, np.pi], description="Phi")
        self.w_radius = FloatSlider(min=0., max=10., step=.1, value=4., description="Radius")

        self.gif_pred = Image(value=b"", format="gif", width=256, height=256)

    def setup_layouts(self) -> None:
        self.setup = GridspecLayout(1, 3)
        self.setup[0, 0] = self.btn_dataset
        self.setup[0, 1] = self.btn_model
        self.setup[0, 2] = self.btn_raymarcher

        self.traj = GridspecLayout(2, 2)
        self.traj[0, 0] = self.w_radius
        self.traj[1, 1] = self.w_theta
        self.traj[1, 0] = self.w_phi
        
        self.viz = GridspecLayout(1, 1)
        self.viz[0, 0] = self.gif_pred

    def setup_dataset(self, change) -> None:
        if hasattr(self, "dataset"):
            del self.dataset

        blender = self.config.blender
        scene = self.config.scene()
        step = self.config.step()
        scale = self.config.scale()

        args = blender, scene, "train"
        self.dataset = BlenderDataset(*args, step=step, scale=scale)

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
        samples = self.config.samples()

        self.raymarcher = BVR(tn, tf, samples=samples)
        
        print("[Setup] Raymarcher Ready")

    def render(self, change) -> None:
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
                self.gif_pred.value = f.read()    

        print("[Setup] Rendering Done ")
        clip = ImageSequenceClip(list(preds), durations=[1. / fps] * frames)
        clip.write_gif(path, fps=fps, verbose=self.verbose)

        with open(path, "rb") as f:
            self.gif_pred.value = f.read()

    def display(self) -> None:
        display(VBox([self.config.app, self.app]))