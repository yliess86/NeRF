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
    from torch.nn import MSELoss
    from torch.optim import AdamW

    
    ROOT = "./data/blender"
    SCENE = "hotdog"
    STEP = 1
    SCALE = .125

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

    LR = 1e-4
    WEIGHT_DECAY = 0

    BATCH_SIZE = 2 ** 10
    JOBS = 24

    EPOCHS = 500
    LOG = 100

    train = BlenderDataset(ROOT, SCENE, "train", step=STEP, scale=SCALE)
    val = BlenderDataset(ROOT, SCENE, "val", step=STEP, scale=SCALE)
    test = BlenderDataset(ROOT, SCENE, "test", step=STEP, scale=SCALE)

    W, H = train.W, train.H
    C, ro, rd = test.C[:W * H], test.ro[:W * H], test.rd[:W * H]

    gt = C.view(W, H, 3).clip(0, 1) * 255
    gt = gt.numpy().astype(np.uint8)
    Image.fromarray(gt).save(GT)

    nerf = NeRF(FEATURES, FEATURES, SIGMA, SIGMA, width=WIDTH, depth=DEPTH).cuda()
    raymarcher = BVR(TN, TF, samples=SAMPLES)
    
    criterion = MSELoss().cuda()
    optim = AdamW(nerf.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    fit_args = raymarcher, optim, criterion, train, None, None  # val, test
    inf_args = raymarcher, ro, rd, W, H

    for _ in range(LOG):
        history = nerf.fit(
            *fit_args,
            epochs=EPOCHS // LOG,
            batch_size=BATCH_SIZE,
            jobs=JOBS,
            perturb=PERTURB,
        )

        torch.save(nerf.state_dict(), MODEL)
        nerf.load_state_dict(torch.load(MODEL))

        pred = nerf.infer(*inf_args, batch_size=BATCH_SIZE)
        pred = pred.numpy().astype(np.uint8)
        Image.fromarray(pred).save(PRED)