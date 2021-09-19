if __name__ == "__main__":
    import nerf.infer
    import nerf.train
    import numpy as np
    import torch
    torch.backends.cudnn.benchmark = True

    from PIL import Image    
    from nerf.data import BlenderDataset
    from nerf.core import NeRF
    from nerf.core import BoundedVolumeRaymarcher as BVR
    from torch.cuda.amp import GradScaler
    from torch.nn import MSELoss
    from torch.optim import AdamW

    
    ROOT = "./data/blender"
    SCENE = "hotdog"
    STEP = 1
    SCALE = .5

    MODEL = f"./res/NeRF_{SCENE}.pt"
    GT = f"./res/NeRF_{SCENE}_gt.png"
    PRED = f"./res/NeRF_{SCENE}_pred.png"

    FEATURES = 256
    SIGMA = 26.

    WIDTH = 256
    DEPTH = 4
    
    TN, TF = 2., 6.
    SAMPLES = 128
    PERTURB = True

    LR = 5e-3
    WEIGHT_DECAY = 1e-2
    FP16 = True

    BATCH_SIZE = 2 ** 10
    JOBS = 24

    EPOCHS = 500
    LOG = 5

    train = BlenderDataset(ROOT, SCENE, "train", step=STEP, scale=SCALE)
    val = BlenderDataset(ROOT, SCENE, "val", step=STEP, scale=SCALE)
    test = BlenderDataset(ROOT, SCENE, "test", step=STEP, scale=SCALE)

    W, H = val.W, val.H
    C, ro, rd = val.C[:W * H], val.ro[:W * H], val.rd[:W * H]

    gt = C.view(W, H, 3).clip(0, 1) * 255
    gt = gt.numpy().astype(np.uint8)
    Image.fromarray(gt).save(GT)

    features, sigma = (FEATURES, FEATURES), (SIGMA, SIGMA)
    nerf = NeRF(*features, *sigma, width=WIDTH, depth=DEPTH).cuda()
    raymarcher = BVR(TN, TF, samples=SAMPLES)
    
    criterion = MSELoss().cuda()
    optim = AdamW(nerf.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(enabled=FP16)

    do_callback = lambda e: (e > 0 and e % (LOG - 1) == 0) or e == (EPOCHS - 1)

    def save_callback(epoch: int) -> None:
        if do_callback(epoch):
            torch.save(nerf.state_dict(), MODEL)

    def render_callback(epoch: int) -> None:
        if do_callback(epoch):
            pred = nerf.infer(raymarcher, ro, rd, W, H, batch_size=BATCH_SIZE)
            pred = pred.numpy().astype(np.uint8)
            Image.fromarray(pred).save(PRED)

    history = nerf.fit(
        raymarcher,
        optim,
        criterion,
        scaler,
        train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        jobs=JOBS,
        perturb=PERTURB,
        callbacks=[save_callback, render_callback]
    )