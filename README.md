# NeRF: Neural Radiance Field

Efficient pytorch implementation of [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934) from Mildenhall et al. 2020.

## Installation

This implementation has been tested on `Ubuntu 20.04` with `Python 3.8`, and `torch 1.9`.
The only requirements are `torch` and `torchvision`.

## Description

NeRF uses both advances in Computer Graphics and Deep Learning research.

The method allows encoding a 3D scene as a continuous volume descibed by density and color at any point in a given bounded volume.
The volume representation is used during volume ray marching as a query of the scene ray intersections.
It is trained in an end-to-end fashion and uses only the ground truth images as an objective signal.
As a way to increase sample efficiency a first network, the coarse model, is trained using voxel grid sampling.
This first pass is used to trained a second network, the fine network, using importance sampling of the volume.

The networks are tied to one unique scene.
Caching and acceleration structures can be used to decrease rendering time during inference.
The same models can be used to generate a depth map and a 3D mesh of the scene.

### Poistional Encoding
### Implicit Representation
### Volume Rendering
### Samplers

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