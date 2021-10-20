import matplotlib.pyplot as plt
import torch.jit as jit

from nerf.core import BoundedVolumeRaymarcher
from nerf.data import BlenderDataset
from nerf.infer import infer
from tqdm import tqdm


dataset = BlenderDataset("data/blender", "lego", "val", step=1, scale=.1)
size = dataset.H * dataset.W

nerf = jit.load("res/lenovo/NeRF_lego.ts")
raymarcher = BoundedVolumeRaymarcher(2., 6.)

fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122, projection="3d")

ax1.set_axis_off()
ax2.set_axis_off()

for i in tqdm(range(100 // dataset.step), desc="Scatter"):
    s = i * size
    e = s + size

    C, ro, rd = dataset.C[s:e], dataset.ro[s:e], dataset.rd[s:e]
    C_ = infer(nerf, raymarcher, ro, rd, dataset.H, dataset.W)
    C_ = C_.view(-1, 3).float() / 255.

    ax1.scatter(ro[0, 0], ro[0, 1], ro[0, 2], c="blue")
    ax2.scatter(ro[0, 0], ro[0, 1], ro[0, 2], c="blue")

    ax1.scatter(rd[:, 0], rd[:, 1], rd[:, 2], c=C)
    ax2.scatter(rd[:, 0], rd[:, 1], rd[:, 2], c=C_)

fig.canvas.draw()
plt.tight_layout(0)
plt.savefig("test.infer.png")