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
        self.root = res
        super().__init__()
        
    def setup_widgets(self) -> None:
        self.register_widget("scene", Dropdown(options=self.scenes, value=self.scenes[3], description="Scene"))
        self.register_widget("step", IntSlider(min=1, max=20, step=1, value=1, description="Step"))
        self.register_widget("scale", FloatSlider(min=.1, max=1., step=.05, value=.5, description="Scale", readout_format=".2f"))
        
        self.register_widget("embedder", Dropdown(options=["PositionalEncoding", "FourrierFeatures"], value="PositionalEncoding", description="Method"))
        self.register_widget("features_x", IntSlider(min=1, max=1024, step=2, value=256, description="X Features"))
        self.register_widget("features_d", IntSlider(min=1, max=1024, step=2, value=64, description="D Features"))
        self.register_widget("sigma_x", FloatSlider(min=1., max=1024., step=1., value=32., description="X Sigma", readout_format=".1f"))
        self.register_widget("sigma_d", FloatSlider(min=1., max=1024., step=1., value=16., description="D Sigma", readout_format=".1f"))
        self.register_widget("freqs_x", IntSlider(min=1, max=128, step=1, value=10, description="X Freqs"))
        self.register_widget("freqs_d", IntSlider(min=1, max=128, step=1, value=4, description="D Freqs"))

        self.register_widget("width", IntSlider(min=32, max=512, step=2, value=128, description="Width"))
        self.register_widget("depth", IntSlider(min=1, max=16, step=1, value=4, description="Depth"))
        self.register_widget("activation", Dropdown(options=["LeakyReLU", "ReLU", "SiLU"], value="SiLU", description="Activation"))
        
        self.register_widget("t", FloatRangeSlider(min=0., max=100., step=1., value=[2., 6.], description="Near-Far", readout_format=".1f"))
        self.register_widget("samples_c", IntSlider(min=8, max=512, step=2, value=64, description="Coarse Samples"))
        self.register_widget("samples_f", IntSlider(min=0, max=512, step=2, value=64, description="Fine Samples"))
        self.register_widget("perturb", Checkbox(value=True, description="Perturb"))
        
        self.register_widget("lr", FloatSlider(min=0., max=1., step=1e-6, value=5e-4, description="Learning Rate", readout_format=".2e"))
        self.register_widget("batch_size", IntSlider(min=2, max=2 ** 14, step=2, value=2 ** 14, description="Batch Size"))
        self.register_widget("jobs", IntSlider(min=0, max=32, step=1, value=cpu_count(), description="Jobs"))
        self.register_widget("log", IntSlider(min=1, max=100, step=1, value=1, description="Log"))

        self.register_widget("strategy", Dropdown(options=["Standard", "Reptile"], value="Reptile", description="Strategy"))
        self.register_widget("fp16", Checkbox(value=True, description="Half Precision"))
        self.register_widget("epochs", IntSlider(min=10, max=100, step=1, value=16, description="Epochs"))
        self.register_widget("steps", IntSlider(min=1, max=64, step=1, value=16, description="Steps"))

        def on_strategy_change(change) -> None:
            self.w_steps.disabled = change.new != "Reptile"

        self.w_steps.disabled = self.strategy() != "Reptile"
        self.w_strategy.observe(on_strategy_change, "value")

        def on_embedder_change(change) -> None:
            self.w_features_x.disabled = change.new != "FourrierFeatures"
            self.w_features_d.disabled = change.new != "FourrierFeatures"
            self.w_sigma_x.disabled = change.new != "FourrierFeatures"
            self.w_sigma_d.disabled = change.new != "FourrierFeatures"
            self.w_freqs_x.disabled = change.new != "PositionalEncoding"
            self.w_freqs_d.disabled = change.new != "PositionalEncoding"

        self.w_features_x.disabled = self.embedder() != "FourrierFeatures"
        self.w_features_d.disabled = self.embedder() != "FourrierFeatures"
        self.w_sigma_x.disabled = self.embedder() != "FourrierFeatures"
        self.w_sigma_d.disabled = self.embedder() != "FourrierFeatures"
        self.w_freqs_x.disabled = self.embedder() != "PositionalEncoding"
        self.w_freqs_d.disabled = self.embedder() != "PositionalEncoding"
        self.w_embedder.observe(on_embedder_change, "value")
        
    def setup_tabs(self) -> None:
        self.register_tab("dataset", 1, 3, ["scene", "step", "scale"])
        self.register_tab("encoding", 4, 2, ["embedder", None, "features_x", "features_d", "sigma_x", "sigma_d", "freqs_x", "freqs_d"])
        self.register_tab("model", 1, 3, ["width", "depth", "activation"])
        self.register_tab("raymarcher", 2, 2, ["t", "perturb", "samples_c", "samples_f"])
        self.register_tab("hyperparams", 2, 2, ["lr", "batch_size", "jobs", "log"])
        self.register_tab("method", 2, 2, ["strategy", "fp16", "epochs", "steps"])
    
    @property
    def res(self) -> str:
        return os.path.join(self.root, self.uuid)

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
