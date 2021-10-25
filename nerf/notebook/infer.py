import gc
import nerf.infer
import numpy as np
import torch
import torch.jit as jit

from IPython.display import display
from ipywidgets.widgets import Button, FloatRangeSlider, Image, Layout, VBox
from moviepy.editor import ImageSequenceClip
from nerf.data import BlenderDataset
from nerf.core import NeRF, BoundedVolumeRaymarcher as BVR
from nerf.notebook.config.infer import InferConfig
from nerf.notebook.core.standard import StandardTabsWidget
from nerf.utils.pbar import tqdm
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

        self.register_widget("theta", FloatRangeSlider(min=-2. * np.pi, max=2. * np.pi, step=.1, value=[-np.pi, np.pi], description="Theta"))
        self.register_widget("phi",   FloatRangeSlider(min=-2. * np.pi, max=2. * np.pi, step=.1, value=[-np.pi, np.pi], description="Phi"))
        self.register_widget("psy",   FloatRangeSlider(min=-2. * np.pi, max=2. * np.pi, step=.1, value=[-np.pi, np.pi], description="Psy"))
        self.register_widget("radius", FloatRangeSlider(min=0., max=10., step=.1, value=[4., 4.], description="Radius"))

        self.register_widget("gif_rgb", Image(value=b"", format="gif", width=256, height=256, layout=Layout(width="80%")))
        self.register_widget("gif_depth", Image(value=b"", format="gif", width=256, height=256, layout=Layout(width="80%")))
        
        self.w_btn_dataset.on_click(self.setup_dataset)
        self.w_btn_model.on_click(self.setup_model)
        self.w_btn_raymarcher.on_click(self.setup_raymarcher)
        self.w_btn_render.on_click(self.render)

        self.w_btn_dataset.disabled = False
        self.w_btn_model.disabled = True
        self.w_btn_raymarcher.disabled = True
        self.w_btn_render.disabled = True

    def setup_tabs(self) -> None:
        self.register_tab("actions", 1, 4, ["btn_dataset", "btn_model", "btn_raymarcher", "btn_render"])
        self.register_tab("traj", 1, 4, ["radius", "theta", "phi", "psy"])
        self.register_tab("viz", 1, 2, ["gif_rgb", "gif_depth"])

    def setup_dataset(self, change) -> None:
        if hasattr(self, "dataset"):
            self.dataset = None

        blender = self.config.blender
        scene = self.config.scene()
        scale = self.config.scale()

        args = blender, scene, "train"
        self.dataset = BlenderDataset(*args, step=None, scale=scale)

        print("[Setup] Dataset Ready")

        self.w_btn_dataset.disabled = True
        self.w_btn_model.disabled = False
        self.w_btn_raymarcher.disabled = True
        self.w_btn_render.disabled = True

    def setup_model(self, change) -> None:
        if hasattr(self, "nerf"):
            self.nerf = None

        path = self.config.model_ts
        self.nerf = jit.load(path).cuda()
        self.nerf.infer = lambda *args, **kwargs: NeRF.infer(self.nerf, *args, **kwargs)
        
        print("[Setup] Model Ready")

        self.w_btn_dataset.disabled = True
        self.w_btn_model.disabled = True
        self.w_btn_raymarcher.disabled = False
        self.w_btn_render.disabled = True

    def setup_raymarcher(self, change) -> None:
        if hasattr(self, "raymarcher"):
            self.raymarcher = None

        tn, tf = self.config.t()
        samples_c = self.config.samples_c()
        samples_f = self.config.samples_f()

        self.raymarcher = BVR(tn, tf, samples_c=samples_c, samples_f=samples_f)
        
        print("[Setup] Raymarcher Ready")

        self.w_btn_dataset.disabled = True
        self.w_btn_model.disabled = True
        self.w_btn_raymarcher.disabled = True
        self.w_btn_render.disabled = False

    def render(self, change) -> None:
        self.config.disable()
        self.disable()

        frames = self.config.frames()
        fps = self.config.fps()
        batch_size = self.config.batch_size()

        theta = self.theta()
        phi = self.phi()
        psy = self.psy()
        radius = self.radius()

        H, W = self.dataset.H, self.dataset.W
        S = H * W

        print("[Render] Creating Rays")
        args = theta, phi, psy, radius, frames
        ros, rds = self.dataset.turnaround_data(*args)
        
        d = next(self.nerf.parameters()).device
        ros, rds = ros.to(d), rds.to(d)

        print("[Render] Rendering Started")
        depth_maps = np.zeros((frames, H, W, 3), dtype=np.uint8)
        rgb_maps = np.zeros((frames, H, W, 3), dtype=np.uint8)
        
        pbar = tqdm(range(0, frames * S, S), desc="[NeRF] Frame", disable=not self.verbose)
        for i, s in enumerate(pbar):
            ro, rd = ros[s:s + S], rds[s:s + S]

            args = self.raymarcher, ro, rd, H, W
            depth_map, rgb_map = self.nerf.infer(*args, batch_size=batch_size, verbose=False)
            
            depth_maps[i] = depth_map.numpy().astype(np.uint8)
            rgb_maps[i] = rgb_map.numpy().astype(np.uint8)

            path = self.config.rgb_map_gif
            PImage.fromarray(rgb_maps[i]).save(path)
            with open(path, "rb") as f:
                self.w_gif_rgb.value = f.read()

            path = self.config.depth_map_gif
            PImage.fromarray(depth_maps[i]).save(path)
            with open(path, "rb") as f:
                self.w_gif_depth.value = f.read()

        print("[Render] Rendering Done")

        print("[Render] GIF rgb_map")
        path = self.config.rgb_map_gif
        clip = ImageSequenceClip(list(rgb_maps), durations=[1. / fps] * frames)
        clip.write_gif(path, fps=fps, verbose=self.verbose)
        
        with open(path, "rb") as f:
            self.w_gif_rgb.value = f.read()

        print("[Render] GIF depth_map")
        path = self.config.depth_map_gif
        clip = ImageSequenceClip(list(depth_maps), durations=[1. / fps] * frames)
        clip.write_gif(path, fps=fps, verbose=self.verbose)
        
        with open(path, "rb") as f:
            self.w_gif_depth.value = f.read()

    def clean(self) -> None:
        self.dataset = None
        self.nerf = None
        self.raymarcher = None

        torch.cuda.empty_cache()
        gc.collect()

        self.config.enable()
        self.enable()

    def display(self) -> None:
        display(VBox([self.config.app, self.app]))