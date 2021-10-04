import os

from ipywidgets.widgets import Checkbox, Dropdown, FloatRangeSlider, FloatSlider, IntSlider
from multiprocessing import cpu_count
from nerf.notebook.config.config import Config
from typing import List


class TrainConfig(Config):
    """Training Config Widget

    Arguments:
        res (str): Path to save all the files
        blender (str): BlenderDataset files path
    """

    def __init__(
        self,
        res: str = "./res",
        blender: str = "./data/blender",
    ) -> None:
        self.blender = blender
        super().__init__()

        self.res = os.path.join(res, self.uuid)
        os.makedirs(self.res, exist_ok=True)
        
    def setup_widgets(self) -> None:
        self.register_widget("scene", Dropdown(options=self.scenes, value=self.scenes[3], description="Scene"))
        self.register_widget("step", IntSlider(min=1, max=20, step=1, value=1, description="Step"))
        self.register_widget("scale", FloatSlider(min=.1, max=1., step=.05, value=1., description="Scale", readout_format=".2f"))
        
        self.register_widget("features", IntSlider(min=32, max=1024, step=2, value=256, description="Features"))
        self.register_widget("sigma", FloatSlider(min=1., max=32., step=1., value=26., description="Sigma", readout_format=".1f"))
        self.register_widget("width", IntSlider(min=32, max=2048, step=2, value=128, description="Width"))
        self.register_widget("depth", IntSlider(min=1, max=8, step=1, value=2, description="Depth"))
        
        self.register_widget("t", FloatRangeSlider(min=0., max=100., step=1., value=[2., 6.], description="Near-Far", readout_format=".1f"))
        self.register_widget("samples_c", IntSlider(min=32, max=512, step=2, value=64, description="Coarse Samples"))
        self.register_widget("samples_f", IntSlider(min=0, max=512, step=2, value=64, description="Fine Samples"))
        self.register_widget("perturb", Checkbox(value=True, description="Perturb"))
        
        self.register_widget("epochs", IntSlider(min=10, max=1_000, step=10, value=100, description="Epochs"))
        self.register_widget("log", IntSlider(min=1, max=100, step=1, value=5, description="Log"))
        self.register_widget("lr", FloatSlider(min=0., max=1., step=1e-6, value=5e-3, description="Learning Rate", readout_format=".2e"))
        self.register_widget("fp16", Checkbox(value=True, description="Half Precision"))
        self.register_widget("batch_size", IntSlider(min=2, max=2 ** 14, step=2, value=2 ** 12, description="Batch Size"))
        self.register_widget("jobs", IntSlider(min=0, max=32, step=1, value=cpu_count() // 2, description="Jobs"))

        self.register_widget("meta", Checkbox(value=False, description="Meta"))
        self.register_widget("meta_steps", IntSlider(min=1, max=100, step=1, value=16, description="Meta Steps", disabled=True))

        def on_meta_change(change) -> None:
            self.w_meta_steps.disabled = not change.new
        
        self.w_meta.observe(on_meta_change, "value")
        
    def setup_tabs(self) -> None:
        self.register_tab("dataset", 2, 2, ["scene", None, "step", "scale"])
        self.register_tab("model", 2, 2, ["features", "sigma", "width", "depth"])
        self.register_tab("raymarcher", 2, 2, ["t", "perturb", "samples_c", "samples_f"])
        self.register_tab("hyperparams", 3, 2, ["epochs", "log", "lr", "fp16", "batch_size", "jobs"])
        self.register_tab("meta learning", 1, 2, ["meta", "meta_steps"])
    
    @property
    def scenes(self) -> List[str]:
        return sorted([
            d for d in os.listdir(self.blender)
            if os.path.isdir(os.path.join(self.blender, d))
        ])

    @property
    def config_yml(self) -> str:
        return os.path.join(self.res, f"NeRF_{self.scene()}_cfg.yml")

    @property
    def model_ts(self) -> str:
        return os.path.join(self.res, f"NeRF_{self.scene()}.ts")
    
    @property
    def gt_png(self) -> str:
        return os.path.join(self.res, f"NeRF_{self.scene()}_gt.png")
        
    @property
    def pred_png(self) -> str:
        return os.path.join(self.res, f"NeRF_{self.scene()}_pred.png")
