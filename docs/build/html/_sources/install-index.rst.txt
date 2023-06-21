*********************
Installation
*********************

The recommended installation is as follows. Create a conda environment and activate it:

.. code-block:: bash

   > conda create -n simDRIFT
   > conda activate simDRIFT

Install the appropriate version of `pytorch <https://pytorch.org>`_ (shown below using the syntax for our lab's GPUs; however, the specific syntax will vary based on your CUDA Toolkit version) :

.. code-block:: bash

   (simDRIFT) > conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

Clone this repository and install simDRIFT:

.. code-block:: bash
     
     (simDRIFT) >  git clone https://github.com/jacobblum/simDRIFT.git
     (simDRIFT) >  pip install -e simDRIFT

To confirm that everything is working as expected, run the test suite:

.. code-block:: bash

     (simDRIFT) > simDRIFT run_tests