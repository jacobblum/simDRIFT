---
title: 'Simulated Diffusion in Realistic Imaging Features of Tissue (Sim-DRIFT)'
tags:
  - Python
  - Diffusion MRI
  - Diffusion Tensor Imaging
  - Biophysics
  - Monte-Carlo Simulation
  - CUDA
authors:
  - name: Jacob Blum
    orcid: 0000-0002-4156-4094
    affiliation: 1
    corresponding: true
  - name: Kainen L. Utt
    orcid: 0000-0002-8555-9000
    affiliation: 1
affiliations:
 - name: Washington University in St. Louis, USA
   index: 1
date: 01 June 2023
bibliography: paper.bib
---

# Summary
This library, `simDRIFT`, provides rapid and flexible Monte-Carlo simulations of Pulsed Gradient Spin Echo (PGSE) diffusion-weighted magnetic resonance imaging (dWI) experiments, which we expect to be useful for dMRI signal processing model development and validation purposes. The primary focus of this library is forward simulations of molecular self-diffusion processes within an ensemble of nuclear magnetic resonance (NMR) active nuclei ("spins") residing in complex, biophysical tissue systems. To acheive a large variety of tissue configurations, `simDRIFT` provides support for $n$ fiber bundles (with user-defined radii, intrinsic diffusivities, orientation angles, and densities) and $m$ cells (with user-defined radii and volume fractions). `simDrift` is written in Python [Python Software Foundation, @VanRossum2010] and supported by a Numba [@Lam2015] backend. Thus, `simDRIFT` benefits from Numba's CUDA API, allowing the simulation of individual spin trajectories to be performed in parallel on single Graphics Processing Unit (GPU) threads. The resulting performance gains support `simDRIFT`'s aim to provide a customizable tool for the rapid prototyping of diffusion models, ground-truth model validation, and in silico phantom production.


# Statement of need
Monte Carlo simulations are particularly effective at generating synthetic diffusion MRI data from complex, biophysically-accurate imaging voxels with known ground-truth microstructural parameters. Consequently, such simulations have proven useful for developing and validating signal processing models [@Chiang2014; @Ye2020]. Existing Monte Carlo simulators, such as CAMINO, Disimpy, Realistic Microstructure Simulator, and others rely on meshes to discretize the computational domain [see, e.g., @Panagiotaki2010, @Yeh2013, @Ianus2016, @Kerkelae2020, @RafaelPatino2020]. While this approach does allow for the representation of complex and finely-detailed microstructural features, given the coarse-graining of such features (fiber bending, etc...) observed at experimentally realistic diffusion times and voxel sizes [@Novikov2018], ``simDRIFT`` supplies a reasonable approximation to the relevant self-diffusion processes by parametizing fibers as narrow cylinders, or "sticks", and cells as isotropic spheres, or "balls" [@Behrens2003]. This way, users may readily simulate voxels induced by multiple, oriented fiber bundles and cells without having to re-discretize the computational domain with complex meshes, which scale in size according to the desired feature resultion, microstructural complexity, and size of the simulated image voxel. 

``simDRIFT``'s superior computational speed represents an important advantage relative to existing Monte Carlo DWI forward simulators. Comparisons between simDRIFT and Disimpy [@Kerkelae2020], a mesh-based, CUDA enabled DWI forward simulator also written in Python, on identical image voxel configurations featuring a 5 $\mu m$ radius cell reveal orders of magnitude improved performance, particularly for large resident spin ensemble sizes. Thus, ``simDRIFT`` is able to very quickly perform large scale simulations, therefore benefiting from favorable convergence properties with respect to the diffusion weighted signal. Indeed, using just desktop or even laptop level GPU hardware, ``simDRIFT`` users are able to quickly and easily generate large amounts of synthetic DWI data from a wide variety of voxel geometires.

![Figure 1: Runtime comparison between simDRIFT and Disimpy, another DWI simulator that runs on the GPU. These simulations were performed on a Windows 10 desktop with an Nvidia RTX 3090 GPU.\label{fig:performance}](figs/simDRIFT_vs_Disimpy.png){ width=70% }

Therefore, the software encompassed by `simDRIFT` fulfills a presently-unmet need by allowing for mesh-free Monte Carlo simulations of dMRI that unify researchers' needs for computational performance and biophysical realism with easy-to-use and highly-configurable open-source software.`simDRIFT` was designed to be used by researchers of all disciplines and focuses who are working with diffusion MRI. Multiple scientific publications which utilize this library are currently in production. The wide customizability, high computational speed, and massively-parallel design will provide avenues for improved model development pipelines and thorough inter-model comparisons, among other potential applications. 

# Features
The library allows users to construct voxel geometries described by user-defined microstructural and scanning parameters. Specifically, `simDRIFT` simulates the diffusion MRI signal generated from the self-diffusion of water in an isotropic imaging voxel of length $L_{\mathrm{voxel}}$ that contains $n \in [0,4]$ distinct fiber bundles and $m \in [0,2]$ distinct cells types, according to the user's selection of diffusion imaging protocol, diffusion time ($\Delta$), time-step size $\mathrm{d}t$, and desired free water diffusivity $D_{FW}$. Within each simulated voxel, the user also has control over the properties of each fiber/cell type. For each fiber bundle, users define the desired orientation (via the angle $\theta$ formed between the bundle and the $z$ axis), intrinsic diffusivity ($D_{i}$), axonal radius ($R_{i}$), and voxel volume fraction $V_{i}/V_{\mathrm{vox}}$. For each cell type, users similarly define the desired radius $R_{j}$ and voxel volume fraction $V_{j}/V_{\mathrm{vox}}$. Examples of such voxel configurations can be seen in \ref{fig:examples}. 


![Figure 2: Example simulated spin trajectories from an imaging voxel featuring two fiber bundles (red, blue) with various orientations ($\theta$ = 0°, 30°, 60°, 90°), along with extra-fiber spins (purple) .\label{fig:examples}](figs/joss_paper_figure.png){ width=70% }

For each time step $\mathrm{d}t$ in the simulation, each tissue compartment’s resident spins are displaced along a randomly chosen direction with a compartment-dependent distance $\mathrm{d}L = \sqrt{6D_{0}dt}$. This process is repeated until the target diffusion time of $\Delta$ is reached. For diffusion times shorter than the expected pre-exchange lifetime of intracellular water, it is safe to assume no exchange between tissue microstructures. The inter-compartmental exchange of water is computationally forbidden via within-timestep rejection of proposed moves beyond the boundaries of each spin’s domain.


# Acknowledgments
This work was funded in part via support from NIH NINDS R01 NS11691 and R01 NS047592. The authors would like to acknowledge the early contributions of Chunyu Song and Anthony Wu. We are also immensely grateful for the supportive and clarifying discussions with Professor Sheng-Kwei Song, whose insight helped clarify this project's trajectory.

# References---
