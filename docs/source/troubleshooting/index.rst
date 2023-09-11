Troubleshooting
===================
If you encounter any issues not covered here, please report them in the `issues section <https://github.com/jacobblum/simDRIFT/issues>`_ of this repository.


Installation 
~~~~~~~~~~~~~~~~~
For certain hardware configurations, we have received notice 
that installation via ``conda`` triggers incompatibilities between
``numba`` and ``llvmlite`` that will cause installation to fail. If you encounter this issue, we reccomend the following installation 
procedure. 

As usual, create a ``conda`` environment and activate it:

.. code-block:: bash

   >conda create -n simDRIFT python=3.8
   >conda activate simDRIFT

Then, install `numba <https://numba.pydata.org/numba-doc/latest/user/installing.html>`_ via ``pip``

.. code-block:: bash
   
  (simDRIFT) >pip install numba==0.56.0

After numba has been installed, please download and install the appropriate `NVIDIA Drivers <https://www.nvidia.com/Download/index.aspx>`_ . Afer the driver installation is complete, we will test the numba install to confirm everything is working. Launch a Python session

.. code-block:: bash
   
  (simDRIFT) >python

Now, type the following commands. If the installation is correct (in the sense that ``numba`` can send data to the GPU), then the output should look something like this:

.. code-block:: bash
   
  >>> import numba
  >>> from numba import cuda
  >>> print(cuda.to_device([1]) 
  <numba.cuda.cudadrv.devicearray.DeviceNDArray object at ....>
  >>> exit()

After this step, installation proceeds as usual. In particular, please install the appropriate version of `pytorch <https://pytorch.org>`_ (shown below using the syntax for our lab's GPUs; however, the specific syntax will vary based on your CUDA Toolkit version) :

.. code-block:: bash
   
  (simDRIFT) >conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia

Clone this repository and install simDRIFT:

.. code-block:: bash
     
     (simDRIFT) >git clone https://github.com/jacobblum/simDRIFT.git
     (simDRIFT) >pip install -e simDRIFT

Finally, to confirm that everything is working as expected, run the test suite:

.. code-block:: bash

     (simDRIFT) >simDRIFT run_tests
