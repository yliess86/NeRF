if __name__ == "__main__":
    import nerf.infer
    import numpy as np
    import torch
    torch.backends.cudnn.benchmark = True

    from moviepy.editor import ImageSequenceClip
    from nerf.data import BlenderDataset
    from nerf.core import NeRF
    from nerf.core import BoundedVolumeRaymarcher as BVR
    from tqdm import tqdm

    
    ROOT = "./data/blender"
    SCENE = "hotdog"
    STEP = 1
    SCALE = 1.

    MODEL = f"./res/NeRF_{SCENE}.pt"
    PRED = f"./res/NeRF_{SCENE}_pred.gif"
    
    FRAMES = 120
    FPS = 25

    FEATURES = 256
    SIGMA = 26.

    WIDTH = 128
    DEPTH = 2
    
    TN, TF = 2., 6.
    SAMPLES = 64
    PERTURB = False

    BATCH_SIZE = 2 ** 14
    JOBS = 24

    data = BlenderDataset(ROOT, SCENE, "val", step=STEP, scale=SCALE)

    ros, rds = data.turnaround_data(samples=FRAMES)
    W, H = data.W, data.H 
    S = W * H
    n = len(ros)

    features, sigma = (FEATURES, FEATURES), (SIGMA, SIGMA)
    nerf = NeRF(*features, *sigma, width=WIDTH, depth=DEPTH).cuda()
    raymarcher = BVR(TN, TF, samples=SAMPLES)

    nerf.load_state_dict(torch.load(MODEL))

    preds = np.zeros((FRAMES, W, H, 3), dtype=np.uint8)
    pbar = tqdm(range(0, n, S), desc="[NeRF] Frame")
    for i, s in enumerate(pbar):
        ro, rd = ros[s:s + S], rds[s:s + S]
        pred = nerf.infer(raymarcher, ro, rd, W, H, batch_size=BATCH_SIZE)
        preds[i] = pred.numpy().astype(np.uint8)
        
    clip = ImageSequenceClip(list(preds), durations=[1. / FPS] * FRAMES)
    clip.write_gif(PRED, fps=FPS)