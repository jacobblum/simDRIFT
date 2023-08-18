
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



@pytest.mark.parametrize("input", [(1.0)])
def test_cell_physics_single(input):
    """1. Check that the forward simulated cell-only signal corresponds to an isotropic diffusion tensor (for only one cell)
    'Note': the inverse problem measured diffusivity here will strongly depend on the diffusion time, thus, this test only requires that the cell diffusion be isotropic    
    """
    bvals, bvecs = diffusion_schemes.get_from_default('DBSI_99')
    gtab = gradient_table(bvals, bvecs)
    tenmodel = dti.TensorModel(gtab)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cmd = r"simDRIFT"
    cmd += f" simulate --n_walkers 256000 --fiber_fractions 0,0 --cell_fractions .1 --cell_radii {input} --Delta 1 --voxel_dims 20 --buffer 0 --verbose no" 
    os.system(cmd)
    signal = signal = nb.load(os.getcwd() + os.sep + 'signals' + os.sep + 'cell_signal.nii').get_fdata()  
    tenfit = tenmodel.fit(signal)
    assert np.isclose(1e3 * tenfit.ad, 1e3 * tenfit.rd, atol = 0.1) 