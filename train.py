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
    from torch.optim import Adam

    
    ROOT = "./data/blender"
    SCENE = "hotdog"
    STEP = 1
    SCALE = .1

    MODEL = f"./res/NeRF_{SCENE}.pt"
    GT = f"./res/NeRF_{SCENE}_gt.png"
    PRED = f"./res/NeRF_{SCENE}_pred.png"

    FEATURES = 256
    SIGMA = 26.

    WIDTH = 128
    DEPTH = 2
    
    TN, TF = 2., 6.
    SAMPLES = 64
    PERTURB = True

    LR = 5e-3
    FP16 = True

    BATCH_SIZE = 2 ** 14
    JOBS = 24

    EPOCHS = 100_000
    LOG = 10

    train = BlenderDataset(ROOT, SCENE, "train", step=STEP, scale=SCALE)
    val = BlenderDataset(ROOT, SCENE, "val", step=STEP, scale=SCALE)
    test = BlenderDataset(ROOT, SCENE, "test", step=STEP, scale=SCALE)

    W, H = val.W, val.H
    ro, rd = val.turnaround_data()
    ro, rd = ro[:W * H], rd[:W * H]

    gt = val.C[:W * H].view(W, H, 3).clip(0, 1) * 255
    gt = gt.numpy().astype(np.uint8)
    Image.fromarray(gt).save(GT)

    features, sigma = (FEATURES, FEATURES), (SIGMA, SIGMA)
    nerf = NeRF(*features, *sigma, width=WIDTH, depth=DEPTH).cuda()
    raymarcher = BVR(TN, TF, samples=SAMPLES)
    
    criterion = MSELoss().cuda()
    optim = Adam(nerf.parameters(), lr=LR)
    scaler = GradScaler(enabled=FP16)

    do_callback = lambda e: (e % (LOG - 1) == 0) or (e == (EPOCHS - 1))

    def save_callback(epoch: int) -> None:
        if do_callback(epoch):
            print(f"[NeRF] Saving Model's Weights to `{MODEL}`")
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