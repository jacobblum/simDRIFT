
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



@pytest.mark.parametrize("input, expected", [((os.path.join(Path(__file__).parents[1], 'src' + os.sep + 'data' + os.sep + 'bval99'), os.path.join(Path(__file__).parents[1], 'src' + os.sep + 'data' + os.sep + 'bvec99')), 99)])
def test_custom_diffusion_scheme(input, expected):
    """1. Check that the forward simulated signal matches the number of bvals and bvecs used in the PGSE experiment w/ 'custom' 
       bvals and bvecs (i.e., loaded in from a path)
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cmd = r"simDRIFT"
    cmd += f" simulate --n_walkers 256000 --fiber_fractions 0,0 --cell_fractions .1,.1 --Delta 0.001 --voxel_dims 20 --buffer 0 --verbose no"
    cmd += f" --bvals {input[0]} --bvecs {input[1]}"
    os.system(cmd)
    signal = nb.load(os.getcwd() + os.sep + 'signals' + os.sep + 'water_signal.nii').get_fdata()  
    assert signal.shape == (expected,)