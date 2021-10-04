import importlib.resources as ires
import os

from . import res
from argparse import ArgumentParser
from socket import gethostbyname, gethostname


with ires.path(res, "nerf.ipynb") as path:
    notebook_dir = os.path.dirname(path)

parser = ArgumentParser("NeRF Notebook")
parser.add_argument("--ip", type=str, default=gethostbyname(gethostname()))
args = parser.parse_args()

cmd = f"jupyter notebook --ip {args.ip} --notebook-dir {notebook_dir}"
os.system(cmd)