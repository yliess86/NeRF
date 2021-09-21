import os

from IPython.display import display
from ipywidgets import GridspecLayout
from ipywidgets.widgets import Checkbox, Dropdown, FloatRangeSlider, FloatSlider, IntSlider, Tab, Widget
from multiprocessing import cpu_count
from typing import Dict, List


class Config:
    def __init__(self, blender: str) -> None:
        self.blender = blender

        self.cfg: List[str] = []
        self.tabs: Dict[str, Widget] = {}

        self.setup_widgets()
        self.setup_tabs()

        self.app = Tab(children=list(self.tabs.values())) 
        for t, title in enumerate(self.tabs.keys()):
            self.app.set_title(t, title.capitalize())

    def register_widget(self, name: str, widget: Widget) -> None:
        setattr(self, f"w_{name}", widget)
        setattr(self, name, lambda: widget.value) 
        self.cfg.append(name)

    def register_tab(self, name: str, rows: int, cols: int, widgets_name: List[str]) -> None:
        self.tabs[name] = grid = GridspecLayout(rows, cols)
        for r in range(rows):
            for c in range(cols):
                if wname := widgets_name[r * cols + c]:
                    grid[r, c] = getattr(self, f"w_{wname}")

    def setup_widgets(self) -> None:
        raise NotImplementedError("'setup' Not Implemented Yet!")

    def setup_tabs(self) -> None:
        raise NotImplementedError("'layout' Not Implemented Yet!")

    def display(self) -> None:
        display(self.app)

    @property
    def scenes(self) -> List[str]:
        return sorted([
            d for d in os.listdir(self.blender)
            if os.path.isdir(os.path.join(self.blender, d))
        ])


class TrainConfig(Config):
    def __init__(self, blender: str = "./data/blender") -> None:
        super().__init__(blender)
        
    def setup_widgets(self) -> None:
        self.register_widget("scene", Dropdown(options=self.scenes, value=self.scenes[3], description="Scene"))
        self.register_widget("step", IntSlider(min=1, max=20, step=1, value=1, description="Step"))
        self.register_widget("scale", FloatSlider(min=.1, max=1., step=.1, value=.1, description="Scale", readout_format=".1f"))
        
        self.register_widget("features", IntSlider(min=32, max=2048, step=2, value=256, description="Features"))
        self.register_widget("sigma", FloatSlider(min=1., max=32., step=1., value=26., description="Sigma", readout_format=".1f"))
        self.register_widget("width", IntSlider(min=32, max=2048, step=2, value=128, description="Width"))
        self.register_widget("depth", IntSlider(min=1, max=8, step=1, value=2, description="Depth"))
        
        self.register_widget("t", FloatRangeSlider(min=0., max=100., step=1., value=[2., 6.], description="Near-Far", readout_format=".1f"))
        self.register_widget("samples", IntSlider(min=32, max=2048, step=2, value=64, description="Samples"))
        self.register_widget("perturb", Checkbox(value=True, description="Perturb"))
        
        self.register_widget("epochs", IntSlider(min=10, max=500_000, step=10, value=100, description="Epochs"))
        self.register_widget("log", IntSlider(min=1, max=100, step=1, value=5, description="Log"))
        self.register_widget("lr", FloatSlider(min=0., max=1., step=1e-6, value=5e-3, description="Learning Rate", readout_format=".2e"))
        self.register_widget("fp16", Checkbox(value=True, description="Half Precision"))
        self.register_widget("batch_size", IntSlider(min=2, max=2 ** 16, step=2, value=2 ** 12, description="Batch Size"))
        self.register_widget("jobs", IntSlider(min=0, max=32, step=1, value=cpu_count() // 2, description="Jobs"))
        
    def setup_tabs(self) -> None:
        self.register_tab("dataset", 2, 2, ["scene", None, "step", "scale"])
        self.register_tab("model", 2, 2, ["features", "sigma", "width", "depth"])
        self.register_tab("raymarcher", 2, 2, ["t", None, "samples", "perturb"])
        self.register_tab("hyperparams", 3, 2, ["epochs", "log", "lr", "fp16", "batch_size", "jobs"])
    
    @property
    def model_ts(self) -> str:
        return f"./res/NeRF_{self.scene()}.ts"
    
    @property
    def gt_png(self) -> str:
        return f"./res/NeRF_{self.scene()}_gt.png"
        
    @property
    def pred_png(self) -> str:
        return f"./res/NeRF_{self.scene()}_pred.png"


class RenderConfig(Config):
    def __init__(self, blender: str = "./data/blender") -> None:
        super().__init__(blender)
    
    def setup_widgets(self) -> None:
        self.register_widget("scene", Dropdown(options=self.scenes, value=self.scenes[3], description="Scene"))
        self.register_widget("step", IntSlider(min=1, max=20, step=1, value=20, description="Step"))
        self.register_widget("scale", FloatSlider(min=.1, max=1., step=.1, value=1., description="Scale", readout_format=".1f"))
        
        self.register_widget("t", FloatRangeSlider(min=0., max=100., step=1., value=[2., 6.], description="Near-Far", readout_format=".1f"))
        self.register_widget("samples", IntSlider(min=32, max=2048, step=2, value=128, description="Samples"))
        self.register_widget("perturb", Checkbox(value=False, description="Perturb", disable=True))
        
        self.register_widget("batch_size", IntSlider(min=2, max=2 ** 16, step=2, value=2 ** 12, description="Batch Size"))
        self.register_widget("jobs", IntSlider(min=0, max=32, step=1, value=cpu_count() // 2, description="Jobs"))

        self.register_widget("frames", IntSlider(min=1, max=500, step=1, value=120, description="Frames"))
        self.register_widget("fps", IntSlider(min=1, max=60, step=1, value=25, description="FPS"))
        
    def setup_tabs(self) -> None:
        self.register_tab("dataset", 2, 2, ["scene", None, "step", "scale"])
        self.register_tab("raymarcher", 2, 2, ["t", None, "samples", "perturb"])
        self.register_tab("hyperparams", 1, 2, ["batch_size", "jobs"])
        self.register_tab("inference", 1, 2, ["frames", "fps"])
    
    @property
    def model_ts(self) -> str:
        return f"./res/NeRF_{self.scene()}.ts"
    
    @property
    def pred_gif(self) -> str:
        return f"./res/NeRF_{self.scene()}_pred.gif"