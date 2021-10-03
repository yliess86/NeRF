import yaml

from nerf.notebook.core.standard import StandardTabsWidget
from uuid import uuid1


class Config(StandardTabsWidget):
    """Config Widget"""

    def __init__(self) -> None:
        super().__init__()
        self.uuid = str(uuid1())

    def save(self, path: str) -> None:
        """Save Config

        Arguments:
            path (str): Path to save the `.yml` file
        """
        data = { name: widget.value for name, widget in self.widgets.items() }
        with open(path, "w") as fp:
            yaml.dump({ "uuid": self.uuid, "data": data }, fp)
    
    def load(self, path: str) -> "Config":
        """Load Config

        Arguments:
            path (str): Path to save the `.yml` file

        Returns:
            self (Config): Current config with laoded values
        """
        with open(path, "r") as fp:
            yml = yaml.load(fp, Loader=yaml.SafeLoader)
        
        self.uuid = yml["uuid"]
        for name, value in yml["data"]:
            self.widgets[name].value = value
        
        return self