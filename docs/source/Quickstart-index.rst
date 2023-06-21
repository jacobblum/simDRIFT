*********************
Quickstart
*********************
Here, we illistrate the basic usage of ``simDRIFT``. 


As a running example, we will operate on the image below.

.. code-block:: python
    
    import os 

    cmd = "simDRIFT"
    cmd += " simulate --n_walkers 256e3 --fiber_fractions .30, .30, .30 --thetas 0, 90, 45"

    os.system(cmd)