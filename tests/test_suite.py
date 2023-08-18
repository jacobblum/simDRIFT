
import sys 
import logging

import os 
import numpy as np
import pytest
import nibabel as nb 
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
import glob as glob
from pathlib import Path
from src.data import diffusion_schemes


import time 


def run_tests():

    log_file = os.path.join(os.getcwd() + os.sep + 'log')
    
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('numpy').setLevel(logging.WARNING)
    logging.basicConfig(level = logging.INFO,
                        format = 'simDRIFT: %(message)s',
                        filename = log_file,
                        filemode = 'w')
    console = logging.StreamHandler()
    formatter = logging.Formatter("simDRIFT: %(message)s")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.info(' ... running the simDRIFT test suite ... ')
    logging.info(' The test suite will run 20 dMRI forward simulations to verify physics, output types, and shapes. On an RTX3090, the test suite takes about 8 minutes to complete; however, this time will depend on your hardware.')

    TEST_SUITE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    logging.info(f' (1/20) Test Signal Types: assert that the forward simulated signal is a Nifti file ')
    cmd = f"pytest {os.path.join(TEST_SUITE_DIR, 'signal_types.py')}"
    os.system(cmd)

    logging.info(f' (2/20) Test Trajectory File Types: assert that the forward trajectory matrix is a .npy file')
    cmd = f"pytest {os.path.join(TEST_SUITE_DIR, 'trajectory_types.py')}"
    os.system(cmd)
    
    logging.info(f'Test Signal Shapes: assert that the forward simulated signal shapes correspond to the input diffusion schemes \n\t  (4/20)-DBSI-99-Direction \n\t  (5/20)-ABCD-103-Direction \n\t  (6/20)-NODDI-145-Direction')
    cmd = f"pytest {os.path.join(TEST_SUITE_DIR, 'signal_shapes.py')}"
    os.system(cmd)

    logging.info(f' (7/20) Test Custom bval/bvec files: assert that the forward simulated signal induced by a custom, input-supplied bvec and bval file matches the shapes from the specified custom diffusion scheme')
    cmd = f"pytest {os.path.join(TEST_SUITE_DIR, 'custom_diffusion_scheme.py')}"
    os.system(cmd)

    logging.info(f'Test Trajectory Shapes: assert that the forward simulated trajectory matrix matches the size of the input number of spins in the ensemble \n\t  (8/20)-100 spins \n\t  (9/20)-256,000 spins \n\t  (10/20)-1,000,000 spins')
    cmd = f"pytest {os.path.join(TEST_SUITE_DIR, 'trajectory_shapes.py')}"
    os.system(cmd)

    
    logging.info(f'Test Water Physics: verify that the forward simulated water-only signal corresponds to a diffusion tensor matching the input water diffusivity parameter, and verify that this diffusion tensor is isotropic \n\t  (11/20)- D_water = 3.0 um^2 / ms <-> AD = RD = 3.0 um^2 / ms  \n\t  (12/20)- D_water = 2.0 um^2 / ms <-> AD = RD = 2.0 um^2 / ms \n\t  (13/20)- D_water = 1.0 um^2 / ms <-> AD = RD = 1.0 um^2 / ms')
    cmd = f"pytest {os.path.join(TEST_SUITE_DIR, 'water_physics.py')}"
    os.system(cmd)

    logging.info(f'Test Fiber Physics: verify that the forward simulated respective fiber-only signals corresponds to a diffusion tensor matching the input fiber diffusivity parameter, and verify that this diffusion tensor is anisotropic \n\t  (14/20)- [D_fiber_1, D_fiber_2, D_fiber_3] = [1.0, 2.0, 2.0] (um^2 /ms) <-> AD >> RD \
                    \n\t  (15/20)- [D_fiber_1, D_fiber_2, D_fiber_3] = [1.0, 1.5, 2.0] (um^2 /ms) <-> AD >> RD\
                    \n\t  (16/20)- [D_fiber_1, D_fiber_2, D_fiber_3] = [1.0, 1.0, 1.5] (um^2 /ms) <-> AD >> RD')
    cmd = f"pytest {os.path.join(TEST_SUITE_DIR, 'fiber_physics.py')}"
    os.system(cmd)

    logging.info(f'Test Cell Physics: verify that the forward simulated cell-only signal, at various cell radii, corresponds to an isotropic diffusion tensor \n\t  (17/20)-r = [1.0 um, 1.0um] <-> AD = RD \n\t  (18/20)-r = [1.5 um, 1.5 um] <-> AD = RD \n\t  (19/20)-r = [2.0 um, 2.0 um] <-> AD = RD')
    cmd = f"pytest {os.path.join(TEST_SUITE_DIR, 'cell_physics.py')}"
    os.system(cmd)

    logging.info(f' (20/20) Test Single Cell Physics: verify that the forward simulated cell-only signal, at various cell radii, corresponds to an isotropic diffusion tensor, r = [1.0 um] <-> AD = RD')
    cmd = f"pytest {os.path.join(TEST_SUITE_DIR, 'single_cell_physics.py')}"
    os.system(cmd)
