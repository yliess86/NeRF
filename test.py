import torch
import torch.jit as jit

from io import BytesIO
from nerf.core import BoundedVolumeRaymarcher as BVR, NeRF
from nerf.core import FourierFeatures as FF, PositionalEncoding as PE
from torch.cuda.amp import autocast, GradScaler
from torch.optim import SGD
from torch.nn import MSELoss


DEVICE = "cuda:0"

n_fms = [FF, PE]
x_fms = [FF(3, 16, 26), PE(3, 6)]
d_fms = [FF(3, 16, 26), PE(3, 6)]

for name, phi_x, phi_d in zip(n_fms, x_fms, d_fms):
    print(f"[Test({name.__name__})] Init")
    nerf = NeRF(phi_x, phi_d, 32, 2).to(DEVICE)
    raymarcher = BVR(2., 6., 16, 16)

    cirterion = MSELoss().to(DEVICE)
    optim = SGD(nerf.parameters(), lr=5e-2, momentum=.9)
    scaler = GradScaler()

    print(f"[Test({name.__name__})] Train")
    nerf = nerf.train()
    for _ in range(2):
        ro = torch.ones((4, 3), requires_grad=False, device=DEVICE)
        rd = torch.ones((4, 3), requires_grad=False, device=DEVICE)
        C = torch.ones((4, 3), requires_grad=False, device=DEVICE)

        with autocast():
            _, C_ = raymarcher.render_volume(nerf, ro, rd, perturb=True, train=True)
            loss = cirterion(C_, C)

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)

    print(f"[Test({name.__name__})] Save")
    buffer = BytesIO()
    jit.save(jit.script(nerf), buffer)

    print(f"[Test({name.__name__})] Load")
    buffer.seek(0)
    nerf = jit.load(buffer).to(DEVICE)

    print(f"[Test({name.__name__})] Inference")
    ro = torch.ones((4, 3), requires_grad=False, device=DEVICE)
    rd = torch.ones((4, 3), requires_grad=False, device=DEVICE)

    nerf = nerf.eval()
    with torch.inference_mode():
        C_ = raymarcher.render_volume(nerf, ro, rd, perturb=True, train=False)
