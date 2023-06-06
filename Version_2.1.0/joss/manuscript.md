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

This library implements a forward simulation of signal acquisition for diffusion-weighted magnetic resonance imaging (dMRI)  complex, biophysically-relevant domains.


# Statement of need

The self-diffusion process of water spins, as measured by nuclear magnetic resonance (NMR), is modeled by the phenomonlogical Bloch-Torrey Equation:
$$\partial_{t} \text{ } \mathbf{M}(\mathbf{r} , t) = \gamma \text{ } \mathbf{M} \times \mathbf{B} -  \frac{\mathbf{M_{x}}\mathbf{\hat{i}} - \mathbf{M_{y}}\mathbf{\hat{j}} }{T_{2}} -\frac{\mathbf{M_{z}}-\mathbf{M_{0}}}{T_{1}}\mathbf{\hat{k}} + \nabla \cdot \mathbf{D}(\mathbf{r})\nabla \mathbf{M}$$
However, analytic solutions to the Bloch-Torrey equation do not exist for complex biophysical systems. To address this problem, we developed this library to provide a framework for biophysically-accurate Monte-Carlo simulations of molecular self-diffusion within biological tissues.


# Related research and software


# Acknowledgments
