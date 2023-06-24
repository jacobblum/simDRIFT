*********************
Installation
*********************
----------------------

The recommended installation is as follows. Create a conda environment and activate it:

.. code-block:: bash

   >conda create -n simDRIFT python=3.8
   >conda activate simDRIFT

Then, install `numba <https://numba.pydata.org/numba-doc/latest/user/installing.html>`_ :  

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

For a quick tutorial on using ``simDRIFT``, please refer to our documentation's `quickstart guide <https://simdrift.readthedocs.io/en/latest/quickstart-index.html>`_.   
