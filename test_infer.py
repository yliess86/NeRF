import matplotlib.pyplot as plt
import torch.jit as jit

from nerf.core import BoundedVolumeRaymarcher
from nerf.data import BlenderDataset
from nerf.infer import infer


dataset = BlenderDataset("data/blender", "lego", "train", step=50, scale=.5)
size = dataset.H * dataset.W

nerf = jit.load("res/issue/NeRF_lego.ts")
raymarcher = BoundedVolumeRaymarcher(2., 6.)

n = 1 * size
C, ro, rd = dataset.C[:n], dataset.ro[:n], dataset.rd[:n]

C_ = infer(nerf, raymarcher, ro, rd, dataset.H, dataset.W)
C_ = C_.view(-1, 3).float() / 255.

fig = plt.figure()
ax = fig.add_subplot(121, projection="3d")

ax.scatter(ro[0, 0], ro[0, 1], ro[0, 2], c="black")
ax.scatter(rd[:, 0], rd[:, 1], rd[:, 2], c=C)

ax = fig.add_subplot(122, projection="3d")

ax.scatter(ro[0, 0], ro[0, 1], ro[0, 2], c="black")
ax.scatter(rd[:, 0], rd[:, 1], rd[:, 2], c=C_)

fig.canvas.draw()
plt.savefig("test.infer.png")