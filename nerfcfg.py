import os

from IPython.display import display
from ipywidgets import GridspecLayout, TwoByTwoLayout, widgets
from typing import List


class TrainConfig:
    def __init__(self) -> None:
        # Dataset
        self.blender = "./data/blender"
        self.scene = self.scenes[3]
        self.step = 1
        self.scale = .1
        
        # Model
        self.features = 256
        self.sigma = 26.
        self.width = 128
        self.depth = 2
        
        # Bounded Volume Raymarcher
        self.tn, self.tf = 2., 6.
        self.samples = 64
        self.perturb = True
        
        # Hyperparameters
        self.epochs = 100
        self.log = 10
        self.lr = 5e-3
        self.fp16 = True
        self.batch_size = 2 ** 14
        self.jobs = 24
        
        # Widgets
        self.w_scene = widgets.Dropdown(options=self.scenes, value=self.scene, description="Scene")
        self.w_step = widgets.IntSlider(min=1, max=20, step=1, value=self.step, description="Step")
        self.w_scale = widgets.FloatSlider(min=.1, max=1., step=.1, value=self.scale, description="Scale")
        
        self.w_features = widgets.IntSlider(min=32, max=2048, step=2, value=self.features, description="Features")
        self.w_sigma = widgets.FloatSlider(min=1., max=32., step=1., value=self.sigma, description="Sigma")
        self.w_width = widgets.IntSlider(min=32, max=2048, step=2, value=self.width, description="Width")
        self.w_depth = widgets.IntSlider(min=1, max=8, step=1, value=self.depth, description="Depth")
        
        self.w_t = widgets.FloatRangeSlider(min=0., max=100., step=1., value=[self.tn, self.tf], description="Near-Far")
        self.w_samples = widgets.IntSlider(min=32, max=2048, step=2, value=self.samples, description="Samples")
        self.w_perturb = widgets.Checkbox(value=self.perturb, description="Perturb")
        
        self.w_epochs = widgets.IntSlider(min=10, max=500_000, step=10, value=self.epochs, description="Epochs")
        self.w_log = widgets.IntSlider(min=1, max=100, step=1, value=self.log, description="Log")
        self.w_lr = widgets.FloatSlider(min=0., max=1., step=1e-6, value=self.lr, description="Learning Rate")
        self.w_fp16 = widgets.Checkbox(value=self.fp16, description="Half Precision")
        self.w_batch_size = widgets.IntSlider(min=2, max=2 ** 16, step=2, value=self.batch_size, description="Batch Size")
        self.w_jobs = widgets.IntSlider(min=0, max=32, step=1, value=self.jobs, description="Jobs")
        
        # Observe Widgets
        self.w_scene.observe(self.update_scene, "value")
        self.w_step.observe(self.update_step, "value")
        self.w_scale.observe(self.update_scale, "value")
        
        self.w_features.observe(self.update_features, "value")
        self.w_sigma.observe(self.update_sigma, "value")
        self.w_width.observe(self.update_width, "value")
        self.w_depth.observe(self.update_depth, "value")
        
        self.w_t.observe(self.update_t, "value")
        self.w_samples.observe(self.update_samples, "value")
        self.w_perturb.observe(self.update_perturb, "value")
        
        self.w_epochs.observe(self.update_epochs, "value")
        self.w_log.observe(self.update_log, "value")
        self.w_lr.observe(self.update_lr, "value")
        self.w_fp16.observe(self.update_fp16, "value")
        self.w_batch_size.observe(self.update_batch_size, "value")
        self.w_jobs.observe(self.update_jobs, "value")
        
        # Widget Layout
        self.w_dataset_title = widgets.HTML(value="<h2>Dataset</h2>", disable=True)
        self.w_dataset_2x2 = TwoByTwoLayout(
            top_left=self.w_scene,
            top_right=None,
            bottom_left=self.w_step,
            bottom_right=self.w_scale,
            merge=False,
        )
        self.w_dataset = widgets.VBox([self.w_dataset_title, self.w_dataset_2x2])
        
        self.w_model_title = widgets.HTML(value="<h2>Model</h2>", disable=True)
        self.w_model_2x2 = TwoByTwoLayout(
            top_left=self.w_features,
            top_right=self.w_sigma,
            bottom_left=self.w_width,
            bottom_right=self.w_depth,
            merge=False,
        )
        self.w_model = widgets.VBox([self.w_model_title, self.w_model_2x2])
        
        self.w_raymarcher_title = widgets.HTML(value="<h2>Raymarcher</h2>", disable=True)
        self.w_raymarcher_2x2 = TwoByTwoLayout(
            top_left=self.w_t,
            top_right=None,
            bottom_left=self.w_samples,
            bottom_right=self.w_perturb,
            merge=False,
        )
        self.w_raymarcher = widgets.VBox([self.w_raymarcher_title, self.w_raymarcher_2x2])
        
        self.w_hyperparams_title = widgets.HTML(value="<h2>Hyperparameters</h2>", disable=True)
        self.w_hyperparams_3x2 = GridspecLayout(3, 2)
        self.w_hyperparams_3x2[0, 0] = self.w_epochs
        self.w_hyperparams_3x2[0, 1] = self.w_log
        self.w_hyperparams_3x2[1, 0] = self.w_lr
        self.w_hyperparams_3x2[1, 1] = self.w_fp16
        self.w_hyperparams_3x2[2, 0] = self.w_batch_size
        self.w_hyperparams_3x2[2, 1] = self.w_jobs
        self.w_hyperparams = widgets.VBox([self.w_hyperparams_title, self.w_hyperparams_3x2])
        
        # App
        self.app_title = widgets.HTML(value="<h1>Train Configuration</h1>", disable=True)
        self.app = widgets.VBox([
            self.app_title,
            self.w_dataset,
            self.w_model,
            self.w_raymarcher,
            self.w_hyperparams,
        ])
    
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
        
    def update_scene(self, change) -> None:
        self.scene = change["new"]
        
    def update_step(self, change) -> None:
        self.step = change["new"]
        
    def update_scale(self, change) -> None:
        self.scale = change["new"]
        
    def update_features(self, change) -> None:
        self.features = change["new"]
        
    def update_sigma(self, change) -> None:
        self.sigma = change["new"]
        
    def update_width(self, change) -> None:
        self.width = change["new"]
        
    def update_depth(self, change) -> None:
        self.depth = change["new"]
        
    def update_t(self, change) -> None:
        self.tn, self.tf = change["new"]
        
    def update_samples(self, change) -> None:
        self.samples = change["new"]
        
    def update_perturb(self, change) -> None:
        self.perturb = change["new"]
        
    def update_epochs(self, change) -> None:
        self.epochs = change["new"]
        
    def update_log(self, change) -> None:
        self.log = change["new"]
        
    def update_lr(self, change) -> None:
        self.lr = change["new"]
        
    def update_fp16(self, change) -> None:
        self.fp16 = change["new"]
        
    def update_batch_size(self, change) -> None:
        self.batch_size = change["new"]
        
    def update_jobs(self, change) -> None:
        self.jobs = change["new"]
        
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
        # Dataset
        self.blender = "./data/blender"
        self.scene = self.scenes[3]
        self.step = 1
        self.scale = .1
        
        # Bounded Volume Raymarcher
        self.tn, self.tf = 2., 6.
        self.samples = 64
        self.perturb = False
        
        # Hyperparameters
        self.batch_size = 2 ** 14
        self.jobs = 24

        # Inference
        self.frames = 120
        self.fps = 25
        
        # Widgets
        self.w_scene = widgets.Dropdown(options=self.scenes, value=self.scene, description="Scene")
        self.w_step = widgets.IntSlider(min=1, max=20, step=1, value=self.step, description="Step")
        self.w_scale = widgets.FloatSlider(min=.1, max=1., step=.1, value=self.scale, description="Scale")
        
        self.w_t = widgets.FloatRangeSlider(min=0., max=100., step=1., value=[self.tn, self.tf], description="Near-Far")
        self.w_samples = widgets.IntSlider(min=32, max=2048, step=2, value=self.samples, description="Samples")
        self.w_perturb = widgets.Checkbox(value=self.perturb, description="Perturb", disable=True)
        
        self.w_batch_size = widgets.IntSlider(min=2, max=2 ** 16, step=2, value=self.batch_size, description="Batch Size")
        self.w_jobs = widgets.IntSlider(min=0, max=32, step=1, value=self.jobs, description="Jobs")

        self.w_frames = widgets.IntSlider(min=1, max=500, step=1, value=self.frames, description="Frames")
        self.w_fps = widgets.IntSlider(min=1, max=60, step=1, value=self.fps, description="FPS")
        
        # Observe Widgets
        self.w_scene.observe(self.update_scene, "value")
        self.w_step.observe(self.update_step, "value")
        self.w_scale.observe(self.update_scale, "value")
        
        self.w_t.observe(self.update_t, "value")
        self.w_samples.observe(self.update_samples, "value")
        
        self.w_batch_size.observe(self.update_batch_size, "value")
        self.w_jobs.observe(self.update_jobs, "value")

        self.w_frames.observe(self.update_frames, "value")
        self.w_fps.observe(self.update_fps, "value")
        
        # Widget Layout
        self.w_dataset_title = widgets.HTML(value="<h2>Dataset</h2>", disable=True)
        self.w_dataset_2x2 = TwoByTwoLayout(
            top_left=self.w_scene,
            top_right=None,
            bottom_left=self.w_step,
            bottom_right=self.w_scale,
            merge=False,
        )
        self.w_dataset = widgets.VBox([self.w_dataset_title, self.w_dataset_2x2])
        
        self.w_raymarcher_title = widgets.HTML(value="<h2>Raymarcher</h2>", disable=True)
        self.w_raymarcher_2x2 = TwoByTwoLayout(
            top_left=self.w_t,
            top_right=None,
            bottom_left=self.w_samples,
            bottom_right=self.w_perturb,
            merge=False,
        )
        self.w_raymarcher = widgets.VBox([self.w_raymarcher_title, self.w_raymarcher_2x2])
        
        self.w_hyperparams_title = widgets.HTML(value="<h2>Hyperparameters</h2>", disable=True)
        self.w_hyperparams_1x2 = GridspecLayout(1, 2)
        self.w_hyperparams_1x2[0, 0] = self.w_batch_size
        self.w_hyperparams_1x2[0, 1] = self.w_jobs
        self.w_hyperparams = widgets.VBox([self.w_hyperparams_title, self.w_hyperparams_1x2])

        self.w_inference_title = widgets.HTML(value="<h2>Inference</h2>", disable=True)
        self.w_inference_1x2 = GridspecLayout(1, 2)
        self.w_inference_1x2[0, 0] = self.w_frames
        self.w_inference_1x2[0, 1] = self.w_fps
        self.w_inference = widgets.VBox([self.w_inference_title, self.w_inference_1x2])
        
        # App
        self.app_title = widgets.HTML(value="<h1>Render Configuration</h1>", disable=True)
        self.app = widgets.VBox([
            self.app_title,
            self.w_dataset,
            self.w_raymarcher,
            self.w_hyperparams,
            self.w_inference,
        ])
    
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
    
    def update_scene(self, change) -> None:
        self.scene = change["new"]
        
    def update_step(self, change) -> None:
        self.step = change["new"]
        
    def update_scale(self, change) -> None:
        self.scale = change["new"]
        
    def update_t(self, change) -> None:
        self.tn, self.tf = change["new"]
        
    def update_samples(self, change) -> None:
        self.samples = change["new"]
        
    def update_batch_size(self, change) -> None:
        self.batch_size = change["new"]
        
    def update_jobs(self, change) -> None:
        self.jobs = change["new"]

    def update_frames(self, change) -> None:
        self.frames = change["new"]

    def update_fps(self, change) -> None:
        self.fps = change["new"]
        
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