# NeRF: Neural Radiance Field

Efficient and comprehensive pytorch implementation of [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934) from Mildenhall et al. 2020.

## Installation

This implementation has been tested on `Ubuntu 20.04` with `Python 3.8`, and `torch 1.9`.
Install required package first `pip3 install -r requirements.txt`.
You may use `pyenv` or `conda` to avoid confilcts with your environement.

## Description

NeRF uses both advances in Computer Graphics and Deep Learning research.

The method allows encoding a 3D scene as a continuous volume described by density and color at any point in a given bounded volume.
During raymarching, the rays query the volume representation model to obtain intersection data.
It is trained in an end-to-end fashion and uses only the ground truth images as an objective signal.
A first network, the coarse model, is trained using voxel grid sampling to increase sample efficiency.
This first pass is used to trained a second network, the fine network, using importance sampling of the volume.

The networks are tied to one unique scene.
Caching and acceleration structures can be used to decrease rendering time during inference.
The same models can be used to generate a depth map and a 3D mesh of the scene.

### Positional Encoding

In [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://arxiv.org/abs/2006.10739), Tancik et al 2020, NeRF authors have shown that encoding positions using fourier feature mapping enables multilayer perceptron to learn high-frequency functions in low dimensional problem domains.

```python
v = xy | xyz                      # normalized to [-1; 1]

rgb = lambda v: mlp(v)            # wo/ ff-encoding
rgb = lambda v: mlp(phi(v))       # w/  ff-encoding

phi = lambda v: [
  a_i * cos(2 * PI * b_i.T * v),
  a_i * sin(2 * PI * b_i.T * v),
].T
```

### Implicit Representation

The scene is encoded by feating a simple multilayer perceptron architecture on density `sigma` and color `RGB` given position `x` and direction `d` queries.

*Original Architecture*
```txt
n = 4

           ReLU    ReLU    
phi(x) --> 256 --> 256 --> ReLU(sigma)
  60    |   n   ^   n  |
        |       |      |           ReLU
        -- cat --      --> 256 --> 128 --> Sigmoid(RGB)
                                ^
                                |
                               cat
                                |
                              phi(d)
                                24
```

### Volume Rendering

Volume raymarching is used to produce the final rendering.
Each ray is thrown from the camera origin to each pixel and sampled `N_c` times for the coarse model and `N_f` times for the fine model between a given bounded volume delimited by the near `t_n` and far `t_f` camera frustum parameters.

*Rendering Equation*
```python
N_c, N_f = 64, 128

alpha_i = (1 - exp(-sigma_i * delta_i))
T_i = cumprod(1 - alpha_i)
w_i = T_i * alpha_i
C_c = sum(w_i * c_i)
```

In this equation, `w_i` respresents a piecewise-constant PDF along the ray, `T_i` the amount of light blocked before reaching segment `t_i`, `delta_i` the segment length `dist(t_i-1, t_i)`, and `c_i` the color of the ray intersection at `t_i`.

The weights `w_i` are reused for inverse transform sampling for the fine pass.
A total of `N_c + N_f` is finally used to generate the last render, this time querying the coarse model instead.


## Implementation (WIP)

*Status (WIP)*
- [x] Fourier Featrure Encoding
- [x] Neural Radiance Field Model
- [x] Bounded Volume Raymarcher
- [x] Noise for Continuous Representation
- [x] Camera Paths (Turnaround, ...)
- [ ] Fine Network
- [ ] Training Optimization
- [ ] Inference Optimization
- [x] Interactive Notebook

*Results (WIP)*
|Ground Truth|Prediction|Inference Animation|
|:----------:|:--------:|:-----------------:|
|![gt](imgs/gt.png)|![pred](imgs/pred.png)|![gif](imgs/pred.gif)|

## Citation

*Original Work*
```txt
@inproceedings{mildenhall2020nerf,
  title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
  author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
  year={2020},
  booktitle={ECCV},
}
```
