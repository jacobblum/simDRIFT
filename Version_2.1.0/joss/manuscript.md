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

This library, simDRIFT, aims to implement a forward simulation of the modelular self-diffusion processes of Nuclear Magtnetic Resonance (NMR) active spins in complex, biophysical tissue systems of various, customizable, configurations. Given the coarse-graining of detailed microstuctural features (fiber bending, etc...) observed at experimentally realistic diffusion times and voxel sizes [Novikov et. al., 2018], simDRIFT represents fibers as narrow cylinders, or "sticks", and cells as isotropic spheres, or "balls" [Behrens et. al., 2003]. We allow users to construct voxel geometries featuring $n$ oriented fiber bundles with chosen intrinsic diffusivities $(D_{0})$ and $m$ cells of various sizes. For each time step $dt$, each tissue compartment’s resident spins are displaced along a randomly chosen direction with a compartment-dependent distance $dL = \sqrt{6D_{0}dt}$. This process is repeated
until the target diffusion time of $\Delta$ is reached. For diffusion times shorter than the expected pre-exchange lifetime of intracellular water, it is safe to assume no exchange between tissue microstructures. The inter-compartmental exchange of water is computationally forbidden in our simulations via within-timestep rejection of proposed moves beyond the boundaries of each spin’s domain. simDRIFT is written in Python and supported by a Numba backend that allows for the simulation of individual spin trajectories to be allocated to a single Graphics Processing Unit (GPU) thread, allowing computation to proceed in an embarssingly parallel manner. In general, simDRIFT provides for rapid and flexible diffusion-weighted Magnetic Resonance Imaging (dMRI) simulations, which we expect to be useful for dMRI signal processing model development and validation purposes.   

# Statement of need

However, analytic solutions to the Bloch-Torrey equation do not exist for complex biophysical systems. To address this problem, we developed this library to provide a framework for biophysically-accurate Monte-Carlo simulations of molecular self-diffusion within biological tissues.


# Related research and software


# Acknowledgments
