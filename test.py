import torch
import torch.jit as jit

from io import BytesIO
from nerf.core import NeRF, BoundedVolumeRaymarcher as BVR
from torch.cuda.amp import autocast, GradScaler
from torch.optim import SGD
from torch.nn import MSELoss


DEVICE = "cuda:0"

nerf = NeRF(16, 16, 26, 26, 32, 2).to(DEVICE)
raymarcher = BVR(2., 6., 16, 16)

cirterion = MSELoss().to(DEVICE)
optim = SGD(nerf.parameters(), lr=5e-2, momentum=.9)
scaler = GradScaler()

nerf = nerf.train()
for _ in range(2):
    ro = torch.zeros((4, 3), requires_grad=False, device=DEVICE)
    rd = torch.zeros((4, 3), requires_grad=False, device=DEVICE)
    C = torch.zeros((4, 3), requires_grad=False, device=DEVICE)

    with autocast():
        C_ = raymarcher.render_volume(nerf, ro, rd, perturb=True, train=True)
        loss = cirterion(C_, C)

    scaler.scale(loss).backward()
    scaler.step(optim)
    scaler.update()
    optim.zero_grad(set_to_none=True)

buffer = BytesIO()
jit.save(jit.script(nerf), buffer)

buffer.seek(0)
nerf = jit.load(buffer).to(DEVICE)

ro = torch.zeros((4, 3), requires_grad=False, device=DEVICE)
rd = torch.zeros((4, 3), requires_grad=False, device=DEVICE)

nerf = nerf.eval()
with torch.inference_mode():
    C_ = raymarcher.render_volume(nerf, ro, rd, perturb=True, train=False)