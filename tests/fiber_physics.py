
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


@pytest.mark.parametrize("input, expected", [((1.0,2.0,2.0), (1.0,2.0,2.0)), ((1.0,1.5,2.0), (1.0,1.5,2.0)), ((1.0,1.0,1.5), (1.0,1.0,1.5))])
def test_fiber_physics_multi(input, expected):
    """1. Check that the forward simulated fiber-only signal corresponds to a diffusion tensor matching the input fiber diffusivities (with multiple fibers)
    2. Check that the forward simulated fiber-only signal corresponds to an anisotropic diffusion tensor (with multiple fibers)
    """
    bvals, bvecs = diffusion_schemes.get_from_default('DBSI_99')
    gtab = gradient_table(bvals, bvecs)
    tenmodel = dti.TensorModel(gtab)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cmd = r"simDRIFT"
    cmd += f" simulate --n_walkers 256000 --fiber_fractions .30,.30,.30 --fiber_diffusions {input[0]},{input[1]},{input[2]} --thetas 0,0,0 --fiber_radii 1,1,1 --cell_fractions 0,0 --Delta 1 --voxel_dims 30 --buffer 0 --verbose no" 
    os.system(cmd)

    fiber_1_signal = nb.load(os.getcwd() + os.sep + 'signals' + os.sep + 'fiber_1_signal.nii').get_fdata()
    fiber_2_signal = nb.load(os.getcwd() + os.sep + 'signals' + os.sep + 'fiber_2_signal.nii').get_fdata()
    fiber_3_signal = nb.load(os.getcwd() + os.sep + 'signals' + os.sep + 'fiber_3_signal.nii').get_fdata()   
    tenfit_1 = tenmodel.fit(fiber_1_signal)
    tenfit_2 = tenmodel.fit(fiber_2_signal)
    tenfit_3 = tenmodel.fit(fiber_3_signal)

    assert (np.isclose(1e3 * np.array([tenfit_1.ad, tenfit_2.ad, tenfit_3.ad]), np.array(expected), atol = 0.1).all())
    assert (np.array([tenfit_1.fa, tenfit_2.fa, tenfit_3.fa]) > .20).all()
