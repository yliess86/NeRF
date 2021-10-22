import os

from ipywidgets.widgets import Checkbox, FloatRangeSlider, FloatSlider, IntSlider
from multiprocessing import cpu_count
from nerf.notebook.config.config import Config
from nerf.notebook.config.train import TrainConfig


class InferConfig(Config):
    """Inference Config Widget

    Arguments:
        blender (str): BlenderDataset files path
    """

    def __init__(
        self,
        tcfg: TrainConfig,
        blender: str = "./data/blender",
    ) -> None:
        self.tcfg = tcfg
        self.blender = blender
        super().__init__()
    
    def setup_widgets(self) -> None:
        self.register_widget("scale", FloatSlider(min=0., max=1., step=.05, value=self.tcfg.scale(), description="Scale", readout_format=".2f"))
        self.register_widget("t", FloatRangeSlider(min=0., max=100., step=1., value=self.tcfg.t(), description="Near-Far", readout_format=".1f"))
        self.register_widget("samples_c", IntSlider(min=32, max=512, step=2, value=self.tcfg.samples_c(), description="Coarse Samples"))
        self.register_widget("samples_f", IntSlider(min=0, max=512, step=2, value=self.tcfg.samples_f(), description="Fine Samples"))
        
        self.register_widget("batch_size", IntSlider(min=2, max=2 ** 14, step=2, value=2 ** 14, description="Batch Size"))
        self.register_widget("jobs", IntSlider(min=0, max=32, step=1, value=cpu_count(), description="Jobs"))

        self.register_widget("frames", IntSlider(min=1, max=500, step=1, value=100, description="Frames"))
        self.register_widget("fps", IntSlider(min=1, max=60, step=1, value=15, description="FPS"))
        
    def setup_tabs(self) -> None:
        self.register_tab("raymarcher", 2, 2, ["scale", "t", "samples_c", "samples_f"])
        self.register_tab("hyperparams", 1, 2, ["batch_size", "jobs"])
        self.register_tab("inference", 1, 2, ["frames", "fps"])

    def scene(self) -> str:
        return self.tcfg.scene()
    
    @property
    def model_ts(self) -> str:
        return self.tcfg.model_ts
    
    @property
    def pred_gif(self) -> str:
        return os.path.join(self.tcfg.res, f"NeRF_{self.scene()}_pred.gif")