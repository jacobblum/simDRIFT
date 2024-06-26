.. image:: https://joss.theoj.org/papers/10.21105/joss.05621/status.svg
   :target: https://doi.org/10.21105/joss.05621

.. image:: https://github.com/jacobblum/dMRI-MCSIM/blob/main/joss/figs/logo.png
  :alt: simDRIFT logo

Introduction
----------------------
This library, ``simDRIFT``, provides for rapid and flexible Monte Carlo simulations of Pulsed Gradient Spin Echo (PGSE) diffusion-weighted magnetic resonance imaging (dMRI) experiments, which we expect to be useful for dMRI signal processing model development and validation purposes. The primary focus of this library is forward simulations of molecular self-diffusion processes within an ensemble of nuclear magnetic resonance (NMR) active nuclei ("spins") residing in complex, biophysical tissue systems. ``simDRIFT`` is written in Python and supported by a Numba backend. Thus, ``simDRIFT`` benefits from Numba's CUDA API, allowing individual spin trajectories to be simulated in parallel on single Graphics Processing Unit (GPU) threads. The resulting performance gains support ``simDRIFT``'s aim to provide a customizable tool for rapidly prototyping diffusion models, ground-truth model validation, and in silico phantom production.

- **Documentation:** https://simdrift.readthedocs.io/en/latest/
- **Source Code:** https://github.com/jacobblum/simDRIFT/tree/main/src
- **Bug Reports:** https://github.com/jacobblum/simDRIFT/issues

Installation
----------------------

Compatibility
~~~~~~~~~~~~~~~~~~~~~
``simDRIFT`` is compatible with Python 3.8 or later, and requires a CUDA device with a compute capability of 3 or higher. We find that in typical use-case simulations on isotropic imaging voxels on the micometer size scale, ``simDRIFT`` will use less than 1.5 Gb of VRAM. For much larger simulations of imaging voxels on the millimeter size scale, typical GPU memory consumption doesn't exceed 2.0 Gb. Thus, we don't anticipate any memory issues given the available memory of compatible GPUs. 

Installing
~~~~~~~~~~~~~~~~~~~~
We recommend installing ``simDRIFT`` in its own conda environment. This allows for easier installation and prevents conflicts with any other Python packages you may have installed. To install ``simDRIFT`` from source:

First, create a conda environment and activate it:

.. code-block:: bash

   >conda create -n simDRIFT python=3.8
   >conda activate simDRIFT

Then, install `numba <https://numba.pydata.org/numba-doc/latest/user/installing.html>`_  by following the linked instructions. For different hardware platforms, the specific numba installation syntax may varry. These instructions are covered in the `numba installation guide <https://numba.pydata.org/numba-doc/latest/user/installing.html>`_. Shown below are the commands for installation on our lab's computers, which are x64-based windows machines. If you encounter any difficulties with this step, please see our `troubleshooting guide <https://simdrift.readthedocs.io/en/latest/troubleshooting/index.html>`_ for an alternative installation procedure. 

.. code-block:: bash
   
  (simDRIFT) >conda install -c numba numba=0.56.0

After numba has been installed, please download and install the appropriate `NVIDIA Drivers <https://www.nvidia.com/Download/index.aspx>`_ . Afer the driver installation is complete, install ``cudatoolkit``:

.. code-block:: bash
   
  (simDRIFT) >conda install cudatoolkit

Also, please install the appropriate version of `pytorch <https://pytorch.org>`_ (shown below using the syntax for our lab's GPUs; however, the specific syntax will vary based on your CUDA Toolkit version) :

.. code-block:: bash
   
  (simDRIFT) >conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia

Clone this repository and install simDRIFT:

.. code-block:: bash
     
     (simDRIFT) >git clone https://github.com/jacobblum/simDRIFT.git
     (simDRIFT) >pip install -e simDRIFT

To confirm that everything is working as expected, run the test suite:

.. code-block:: bash

     (simDRIFT) >simDRIFT run_tests

Development
----------------
To contribute to ``simDRIFT`` , or to seek support and report any issues or problems with the software, please follow the instructions laid out `here <https://github.com/jacobblum/simDRIFT/blob/main/CONTRIBUTING.md>`_!
