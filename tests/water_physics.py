
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

@pytest.mark.parametrize("input, expected", [(3.0, 3.0), (1.0, 1.0), (2.0, 2.0)])
def test_water_physics(input, expected):
    """1. Check that the forward simulated water-only signal corresponds to a diffusion tensor matching the input water diffusivity
       2. Check that the forward simulated water-only signal corresponds to an isotropic diffusion tensor 
    """

    bvals, bvecs = diffusion_schemes.get_from_default('DBSI_99')
    gtab = gradient_table(bvals, bvecs)
    tenmodel = dti.TensorModel(gtab)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cmd = r"simDRIFT"
    cmd += f" simulate --n_walkers 256000 --fiber_fractions 0,0 --cell_fractions 0,0 --Delta 1 --water_diffusivity {input} --voxel_dims 10 --buffer 0 --verbose no" 
    os.system(cmd)
    signal = nb.load(os.getcwd() + os.sep + 'signals' + os.sep + 'water_signal.nii').get_fdata()  
    tenfit = tenmodel.fit(signal)
    assert np.isclose(1e3 * tenfit.ad, expected, atol = .1) 
    assert np.isclose(1e3 * tenfit.ad, 1e3 * tenfit.rd, atol=.1)
