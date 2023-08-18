
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



@pytest.mark.parametrize("input, expected", [('DBSI_99', 99), ('ABCD', 103), ('NODDI_145', 145)])
def test_signal_shapes(input, expected):
    """1. Check that the forward simulated signal matches the number of bvals and bvecs used in the 'imaging' experiment 
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cmd = r"simDRIFT"
    cmd += f" simulate --n_walkers 256000 --fiber_fractions 0,0 --cell_fractions 0,0 --Delta 1 --diff_scheme {input} --voxel_dims 10 --buffer 0 --verbose no" 
    os.system(cmd)
    signal = nb.load(os.getcwd() + os.sep + 'signals' + os.sep + 'water_signal.nii').get_fdata()  
    assert signal.shape == (expected,)