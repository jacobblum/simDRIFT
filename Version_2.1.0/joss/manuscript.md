---
title: 'Simulated Diffusion in Realistic Imagaing Features of Tissue (Sim-DRIFT)'
tags:
  - Python
  - Diffusion MRI
  - Diffusion Tensor Imaging
  - Monte-Carlo Simulation
  - CUDA
authors:
  - name: Jacob Blum
    orcid: 0000-0002-4156-4094
    affiliation: 1
  - name: Kainen L. Utt
    orcid: 0000-0002-8555-9000
    affiliation: 1
affiliations:
 - name: Washington University in St. Louis, St. Louis, MO, USA
   index: 1
date: 01 June 2023
---

# Summary
This library, `simDRIFT`, provides rapid and flexible Monte-Carlo simulations of diffusion-weighted Magnetic Resonance Imaging (dMRI), which we expect to be useful for dMRI signal processing model development and validation purposes. In particular, we aim to implement a forward simulation of the modular self-diffusion processes of an ensemble of Nuclear Magnetic Resonance (NMR) active spins in complex, biophysical tissue systems of various, customizable, configurations admitted by $n$ oriented fiber bundles of set-able intrinsic diffusivities and $m$ cells. simDrift is written in Python (Python Software Foundation) and supported by a Numba (Lam, Pitrou & Siebert, 2015) backend. Thus, simDRIFT benefits Numba's CUDA API, allowing for the simulation of individual spin trajectories to be performed in an embarrassingly parallel manner on single Graphics Processing Unit (GPU) threads.



# Statement of need
Owing to its ability to generate synthetic dMRI data admitted by complex, biophysically accurate imaging voxels with known ground-truth microstructural parameters, Monte-Carlo simulation of dMRI has demonstrated usefulness in signal processing model development and validation (Chiang, 2014, Ye, 2021). Existing Monte-Carlo simulators typically rely on meshes to discretize the computational domain. While this does allow for the representation of complex, finely detailed microstructural elements, creating meshes for the biologically relevant, complex 3D geometries found in typical imaging voxels can be difficult, and may represent a barrier of use to researchers who lack numerics experience and training. The software encompassed by `simDRIFT` therefore fulfills a thus far unmet by allowing for mesh-free dMRI Monte Carlo simulation that unifies researchers needs for computational performance and biophysical realism with an easy to use and highly configurable software.

# Features
Given the coarse-graining of detailed microstructural features (fiber bending, etc...) observed at experimentally realistic diffusion times and voxel sizes (Novikov et. al., 2018), `simDRIFT` represents fibers as narrow cylinders, or "sticks", and cells as isotropic spheres, or "balls" (Behrens et. al., 2003). We allow users to construct voxel geometries featuring $n$ oriented fiber bundles with chosen intrinsic diffusivities $(D_{0})$ and $m$ cells of various sizes [Figure 1]. 

<p align = "center"> <img src = "https://github.com/jacobblum/dMRI-MCSIM/blob/dev/Version_2.1.0/joss/figs/simulation_configuration.png" width = "650" height = "500"> </p> 
<p align = "center"> Figure 1: Example simulated spin trajectories from an imaging voxel featuring two fiber bundles (red, blue) with various orientations ($\theta = 0^{\circ}, 30^{\circ}, 60^{\circ}, 90^{\circ}$), along with cells (green) </p> 



For each time step ($dt$), each tissue compartment’s resident spins are displaced along a randomly chosen direction with a compartment-dependent distance $dL = \sqrt{6D_{0}dt}$. This process is repeated until the target diffusion time of Δ is reached. For diffusion times shorter than the expected pre-exchange lifetime of intracellular water, it is safe to assume no exchange between tissue microstructures. The inter-compartmental exchange of water is computationally forbidden via within-timestep rejection of proposed moves beyond the boundaries of each spin’s domain.


# Acknowledgments

