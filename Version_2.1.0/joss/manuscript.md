---
title: 'Title HERE'
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
  - name: Kainen Utt
    orcid: 0000-0002-8555-9000
    affiliation: 1
affiliations:
 - name: Washington University in St. Louis, St. Louis, MO, USA
   index: 1
date: 01 June 2023
---

# Summary

This library implements a forward simulation of diffusion-weighted Magnetic Resonance Imaging (dMRI) signal aquisition on complex, biologically relevant domains.


# Statement of need

The self-diffusion processes of Nuclear Magnetic Resonance (NMR) spins is modeled by the phenomonlogical Bloch-Torrey Equation:
$$\partial_{t} \text{ } \mathbf{M}(\mathbf{r} , t) = \gamma \text{ } \mathbf{M} \times \mathbf{B} -  \frac{\mathbf{M_{x}}\mathbf{\hat{i}} - \mathbf{M_{y}}\mathbf{\hat{j}} }{T_{2}} -\frac{\mathbf{M_{z}}-\mathbf{M_{0}}}{T_{1}}\mathbf{\hat{k}} + \nabla \cdot \mathbf{D}(\mathbf{r})\nabla \mathbf{M}$$
However, for complex, bioloigcally relevant domains, analytic solutions to Bloch-Torrey do not exist. This library was developed to implement Monte-Carlo simulation of molecular self diffusion processes in complex biological domains.



# Related research and software


# Acknowledgments
