.. image:: https://github.com/jacobblum/dMRI-MCSIM/blob/main/joss/figs/logo.png
  :alt: simDRIFT logo

This library, ``simDRIFT``, provides rapid and flexible Monte-Carlo simulations of diffusion-weighted magnetic resonance imaging (dMRI), which we expect to be useful for dMRI signal processing model development and validation purposes. The primary focus of this library is forward simulations of modular self-diffusion processes within an ensemble of nuclear magnetic resonance (NMR) active nuclei ("spins") residing in complex, biophysical tissue systems. ``simDrift`` is written in Python and supported by a Numba backend. Thus, ``simDRIFT`` benefits from Numba's CUDA API, allowing individual spin trajectories to be simulated in parallel on single Graphics Processing Unit (GPU) threads. The resulting performance gains support ``simDRIFT``'s aim to provide a customizable tool for rapidly prototyping diffusion models, ground-truth model validation, and in silico phantom production.

The current release contains the following modules:

* ``simulate``

  this module supports $n$ fiber bundles (with user-defined radii, intrinsic diffusivities, orientation angles, and densities) and   
  $m$ cells (with user-defined radii and volume fractions). Given the coarse-graining of detailed microstructural features (fiber bending, 
  etc...) observed at experimentally realistic diffusion times and voxel sizes, `simDRIFT` represents fibers as narrow cylinders, 
  or "sticks", and cells as isotropic spheres, or "balls". The module executes the forward diffusion MRI simulation on voxel geometries described
  by user-defined microstructural and scanning parameters. 
  A quick start tutorial can be found
  `here <https://en.wikipedia.org/wiki/Bloch_equations>`_.

* ``run_tests``

  this module runs the test suite for ``simDRIFT``

Installation and Usage
----------------------

Manual installation
~~~~~~~~~~~~~~~~~~~

The recommended installation is as follows. Create a conda environment and activate it:

.. code-block:: bash

   > conda create -n simDRIFT
   > conda activate simDRIFT

Install the appropriate version of `pytorch <https://pytorch.org>`_ (shown below using the syntax for our lab's GPUs; however, the specific syntax will vary based on your CUDA Toolkit version) :

.. code-block:: bash

   (simDRIFT) > conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

Clone this repository and install simDRIFT:

.. code-block:: bash
     
     (simDRIFT) >  git clone https://github.com/jacobblum/dMRI-MCSIM.git
     (simDRIFT) >  pip install -e simDRIFT

To confirm that everything is working as expected, run the test suite:

.. code-block:: bash

     (simDRIFT) > simDRIFT run_tests


Citing simDRIFT
-----------------

If you use simDRIFT in your research, please cite our associated JOSS paper. 
