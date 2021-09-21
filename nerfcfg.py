import os

from IPython.display import display
from ipywidgets import GridspecLayout, TwoByTwoLayout, widgets
from multiprocessing import cpu_count
from typing import List


class TrainConfig:
    def __init__(self) -> None:
        self.blender = "./data/blender"
        
        # Widgets
        self.w_scene = widgets.Dropdown(options=self.scenes, value=self.scenes[3], description="Scene")
        self.w_step = widgets.IntSlider(min=1, max=20, step=1, value=1, description="Step")
        self.w_scale = widgets.FloatSlider(min=.1, max=1., step=.1, value=.1, description="Scale", readout_format=".1f")
        
        self.w_features = widgets.IntSlider(min=32, max=2048, step=2, value=256, description="Features")
        self.w_sigma = widgets.FloatSlider(min=1., max=32., step=1., value=26., description="Sigma", readout_format=".1f")
        self.w_width = widgets.IntSlider(min=32, max=2048, step=2, value=128, description="Width")
        self.w_depth = widgets.IntSlider(min=1, max=8, step=1, value=2, description="Depth")
        
        self.w_t = widgets.FloatRangeSlider(min=0., max=100., step=1., value=[2., 6.], description="Near-Far", readout_format=".1f")
        self.w_samples = widgets.IntSlider(min=32, max=2048, step=2, value=64, description="Samples")
        self.w_perturb = widgets.Checkbox(value=True, description="Perturb")
        
        self.w_epochs = widgets.IntSlider(min=10, max=500_000, step=10, value=100, description="Epochs")
        self.w_log = widgets.IntSlider(min=1, max=100, step=1, value=5, description="Log")
        self.w_lr = widgets.FloatSlider(min=0., max=1., step=1e-6, value=5e-3, description="Learning Rate", readout_format=".2e")
        self.w_fp16 = widgets.Checkbox(value=True, description="Half Precision")
        self.w_batch_size = widgets.IntSlider(min=2, max=2 ** 16, step=2, value=2 ** 12, description="Batch Size")
        self.w_jobs = widgets.IntSlider(min=0, max=32, step=1, value=cpu_count(), description="Jobs")
        
        # Widget Layout
        self.w_dataset = TwoByTwoLayout(
            top_left=self.w_scene,
            top_right=None,
            bottom_left=self.w_step,
            bottom_right=self.w_scale,
            merge=False,
        )
        
        self.w_model = TwoByTwoLayout(
            top_left=self.w_features,
            top_right=self.w_sigma,
            bottom_left=self.w_width,
            bottom_right=self.w_depth,
            merge=False,
        )
        
        self.w_raymarcher = TwoByTwoLayout(
            top_left=self.w_t,
            top_right=None,
            bottom_left=self.w_samples,
            bottom_right=self.w_perturb,
            merge=False,
        )
        
        self.w_hyperparams = GridspecLayout(3, 2)
        self.w_hyperparams[0, 0] = self.w_epochs
        self.w_hyperparams[0, 1] = self.w_log
        self.w_hyperparams[1, 0] = self.w_lr
        self.w_hyperparams[1, 1] = self.w_fp16
        self.w_hyperparams[2, 0] = self.w_batch_size
        self.w_hyperparams[2, 1] = self.w_jobs
        
        # App
        self.app_title = widgets.HTML(value="<h1>Train Configuration</h1>", disable=True)
        self.app_tabs = widgets.Tab()
        self.app_tabs.children = [self.w_dataset, self.w_model, self.w_raymarcher, self.w_hyperparams] 
        for t, title in enumerate(["Dataset", "Model", "Raymarcher", "Hyperparameters"]):
            self.app_tabs.set_title(t, title)
        self.app = widgets.VBox([self.app_title, self.app_tabs])
    
    @property
    def scenes(self) -> List[str]:
        return sorted([
            d for d in os.listdir(self.blender)
            if os.path.isdir(os.path.join(self.blender, d))
        ])
    
    @property
    def model_ts(self) -> str:
        return f"./res/NeRF_{self.scene}.ts"
    
    @property
    def gt_png(self) -> str:
        return f"./res/NeRF_{self.scene}_gt.png"
        
    @property
    def pred_png(self) -> str:
        return f"./res/NeRF_{self.scene}_pred.png"
        
    @property
    def scene(self) -> str:
        return self.w_scene.value
    
    @property
    def step(self) -> int:
        return self.w_step.value
    
    @property
    def scale(self) -> float:
        return self.w_scale.value
    
    @property
    def features(self) -> int:
        return self.w_features.value
    
    @property
    def sigma(self) -> float:
        return self.w_sigma.value
    
    @property
    def width(self) -> int:
        return self.w_width.value
    
    @property
    def depth(self) -> int:
        return self.w_depth.value
    
    @property
    def tn(self) -> float:
        return self.w_t.value[0]
    
    @property
    def tf(self) -> float:
        return self.w_t.value[1]
    
    @property
    def samples(self) -> int:
        return self.w_samples.value
    
    @property
    def perturb(self) -> bool:
        return self.w_perturb.value
    
    @property
    def epochs(self) -> int:
        return self.w_epochs.value
    
    @property
    def log(self) -> int:
        return self.w_log.value
    
    @property
    def lr(self) -> float:
        return self.w_lr.value
    
    @property
    def fp16(self) -> bool:
        return self.w_fp16.value
    
    @property
    def batch_size(self) -> int:
        return self.w_batch_size.value
    
    @property
    def jobs(self) -> int:
        return self.w_jobs.value
        
    def display(self) -> None:
        display(self.app)

    def __str__(self) -> str:
        return f"""TrainConfig:
    # Dataset
    {self.blender = }
    {self.scene =}
    {self.step = }
    {self.scale = }
    
    # Model
    {self.features = }
    {self.sigma = }
    {self.width = }
    {self.depth = }
    
    # Bounded Volume Raymarcher
    {self.tn = }, {self.tf = }
    {self.samples = }
    {self.perturb = }
    
    # Hyperparameters
    {self.epochs = }
    {self.log = }
    {self.lr = }
    {self.fp16 = }
    {self.batch_size = }
    {self.jobs = }

    # Paths
    {self.model_ts = }
    {self.gt_png = }
    {self.pred_png = }
        """


class RenderConfig:
    def __init__(self) -> None:
        self.blender = "./data/blender"
        
        # Widgets
        self.w_scene = widgets.Dropdown(options=self.scenes, value=self.scenes[3], description="Scene")
        self.w_step = widgets.IntSlider(min=1, max=20, step=1, value=20, description="Step")
        self.w_scale = widgets.FloatSlider(min=.1, max=1., step=.1, value=1., description="Scale", readout_format=".1f")
        
        self.w_t = widgets.FloatRangeSlider(min=0., max=100., step=1., value=[2., 6.], description="Near-Far", readout_format=".1f")
        self.w_samples = widgets.IntSlider(min=32, max=2048, step=2, value=128, description="Samples")
        self.w_perturb = widgets.Checkbox(value=False, description="Perturb", disable=True)
        
        self.w_batch_size = widgets.IntSlider(min=2, max=2 ** 16, step=2, value=2 ** 12, description="Batch Size")
        self.w_jobs = widgets.IntSlider(min=0, max=32, step=1, value=cpu_count(), description="Jobs")

        self.w_frames = widgets.IntSlider(min=1, max=500, step=1, value=120, description="Frames")
        self.w_fps = widgets.IntSlider(min=1, max=60, step=1, value=25, description="FPS")
        
        # Widget Layout
        self.w_dataset = TwoByTwoLayout(
            top_left=self.w_scene,
            top_right=None,
            bottom_left=self.w_step,
            bottom_right=self.w_scale,
            merge=False,
        )
        
        self.w_raymarcher = TwoByTwoLayout(
            top_left=self.w_t,
            top_right=None,
            bottom_left=self.w_samples,
            bottom_right=self.w_perturb,
            merge=False,
        )
        
        self.w_hyperparams = GridspecLayout(1, 2)
        self.w_hyperparams[0, 0] = self.w_batch_size
        self.w_hyperparams[0, 1] = self.w_jobs

        self.w_inference = GridspecLayout(1, 2)
        self.w_inference[0, 0] = self.w_frames
        self.w_inference[0, 1] = self.w_fps
        
        # App
        self.app_title = widgets.HTML(value="<h1>Render Configuration</h1>", disable=True)
        self.app_tabs = widgets.Tab()
        self.app_tabs.children = [self.w_dataset, self.w_raymarcher, self.w_hyperparams, self.w_inference] 
        for t, title in enumerate(["Dataset", "Raymarcher", "Hyperparameters", "Inference"]):
            self.app_tabs.set_title(t, title)
        self.app = widgets.VBox([self.app_title, self.app_tabs])
    
    @property
    def scenes(self) -> List[str]:
        return sorted([
            d for d in os.listdir(self.blender)
            if os.path.isdir(os.path.join(self.blender, d))
        ])
    
    @property
    def model_ts(self) -> str:
        return f"./res/NeRF_{self.scene}.ts"
    
    @property
    def pred_gif(self) -> str:
        return f"./res/NeRF_{self.scene}_pred.gif"
    
    @property
    def scene(self) -> str:
        return self.w_scene.value
        
    @property
    def step(self) -> int:
        return self.w_step.value
        
    @property
    def scale(self) -> float:
        return self.w_scale.value
        
    @property
    def tn(self) -> float:
        return self.w_t.value[0]

    @property
    def tf(self) -> float:
        return self.w_t.value[1]
        
    @property
    def samples(self) -> int:
        return self.w_samples.value

    @property
    def perturb(self) -> bool:
        return self.w_perturb.value
        
    @property
    def batch_size(self) -> int:
        return self.w_batch_size.value
        
    @property
    def jobs(self) -> int:
        return self.w_jobs.value

    @property
    def frames(self) -> int:
        return self.w_frames.value

    @property
    def fps(self) -> int:
        return self.w_fps.value
        
    def display(self) -> None:
        display(self.app)

    def __str__(self) -> str:
        return f"""RenderConfig:
    # Dataset
    {self.blender = }
    {self.scene = }
    {self.step = }
    {self.scale = }
    
    # Bounded Volume Raymarcher
    {self.tn = }, {self.tf = }
    {self.samples = }
    {self.perturb = }
    
    # Hyperparameters
    {self.batch_size = }
    {self.jobs = }

    # Inference
    {self.frames = }
    {self.fps = }

    # Paths
    {self.model_ts = }
    {self.pred_gif = }
        """