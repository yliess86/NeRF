from IPython.display import display
from ipywidgets import GridspecLayout
from ipywidgets.widgets import Button, Tab, Widget
from typing import Dict, List


class StandardTabsWidget:
    """Standard Tabs Widget

    Helper to register widgets in tabs in a friendly way.
    """

    def __init__(self) -> None:
        self.widgets: Dict[str, Widget] = {}
        self.tabs: Dict[str, Widget] = {}

        self.setup_widgets()
        self.setup_tabs()

        self.app = Tab(children=list(self.tabs.values())) 
        for t, title in enumerate(self.tabs.keys()):
            self.app.set_title(t, title.capitalize())

    def register_widget(self, name: str, widget: Widget) -> None:
        """Register a Widget and its Value Property
        
        Widgets are accesed via self.w_[name]
        and their property by self.[name]

        Arguments:
            name (str): Widget name
            widget (Widget): Widget Definition
        """
        if isinstance(widget, Button):
            setattr(widget, "value", None)

        setattr(self, f"w_{name}", widget)
        setattr(self, name, lambda: widget.value)
        self.widgets[name] = getattr(self, f"w_{name}")

    def register_tab(
        self,
        name: str,
        rows: int,
        cols: int,
        widgets_name: List[str],
    ) -> None:
        """Register Tabs and their Widgets in Grid Layouts

        Arguments:
            name (str): Tab name
            rows (int): Number of rows
            cols (int): Number of columns
            widgets_name (List[str]): Widgets name in order
        """
        self.tabs[name] = grid = GridspecLayout(
            rows,
            cols,
            justify_content="center",
            align_items="center",
        )

        for r in range(rows):
            for c in range(cols):
                if wname := widgets_name[r * cols + c]:
                    grid[r, c] = getattr(self, f"w_{wname}")

    def enable_widget(self, name: str) -> None:
        """Enable Widget given Name
        
        Arguments:
            name (str): Widget name
        """
        if name in self.widgets:
            self.widgets[name].disabled = False
    
    def disable_widget(self, name: str) -> None:
        """Disable Widget given Name
        
        Arguments:
            name (str): Widget name
        """
        if name in self.widgets:
            self.widgets[name].disabled = True

    def enable(self) -> None:
        """Enable all Widgets"""
        for widget in self.widgets.values():
            widget.disabled = False

    def disable(self) -> None:
        """Disable all Widgets"""
        for widget in self.widgets.values():
            widget.disabled = True

    def setup_widgets(self) -> None:
        raise NotImplementedError("'setup' Not Implemented Yet!")

    def setup_tabs(self) -> None:
        raise NotImplementedError("'layout' Not Implemented Yet!")

    def display(self) -> None:
        """Call to display Application's Widget in a Notebook"""
        display(self.app)