What is simDRIFT 
===================
``simDRIFT`` is a software package for massively parallel forward simulation of diffusion weighted MRI on biophysically mimicking tissue systems

Scope and Purpose
-------------------

This library, ``simDRIFT``, provides rapid and flexible Monte-Carlo simulations of Pulsed Gradient Spin Echo (PGSE) diffusion-weighted magnetic resonance imaging (DWI) experiments, which we expect to be useful for DWI signal processing model development and validation purposes. The primary focus of this library is forward simulations of molecular self-diffusion processes within an ensemble of nuclear magnetic resonance (NMR) active nuclei ("spins") residing in complex, biophysical tissue systems. To achieve a large variety of tissue configurations, simDRIFT provides support for fiber bundles (with user-defined radii, intrinsic diffusivities, orientations, and densities) and cells.

Monte Carlo simulations are particularly effective at generating synthetic DWI data from complex, biophysically-mimicking image voxels with known ground-truth microstructural parameters. Consequently, such simulations have proven useful for developing and validating signal processing models. Existing Monte Carlo simulators typically rely on meshes to discretize the computational domain. While this approach does allow for the representation of complex and finely detailed microstructural elements, creating meshes for the biologically relevant 3D geometries found in typical image voxels can be difficult and may therefore present a barrier to wide use among researchers who lack experience and training in computational mathematics. The software encompassed by ``simDRIFT`` therefore fulfills a presently unmet need by allowing for mesh-free Monte Carlo simulations of DWI that unify researchers' needs for computational performance and biophysical realism with easy-to-use and highly configurable software.

``simDRIFT`` was designed to be used by researchers of all disciplines and focuses who are working with diffusion MRI (dMRI). Multiple scientific publications which utilize this library are currently in production. The wide customizability, high computational speed, and massively parallel design will provide avenues for improved model development pipelines and thorough inter-model comparisons, among other potential applications. These same traits may also make simDRIFT useful for instructors of signal processing or diffusion imaging courses.

Modules
-----------
The current release contains the following modules:

* ``simulate``

  this module supports :math:`n` fiber bundles (with user-defined radii, intrinsic diffusivities, orientations, and densities) and   
  :math:`m` cells (with user-defined radii and volume fractions). Given the coarse graining of detailed microstructural features (fiber bending, 
  etc...) observed at experimentally comperable diffusion times and voxel sizes, ``simDRIFT`` represents fibers as narrow cylinders, 
  or "sticks", and cells as isotropic spheres, or "balls". The module executes the forward DWI simulation on voxel geometries described
  by user defined microstructural and scanning parameters. 

* ``run_tests``

  this module runs the test suite for ``simDRIFT``
