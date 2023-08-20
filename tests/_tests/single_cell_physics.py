
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
import configparser


@pytest.mark.parametrize("input", [(1.0)])
def test_cell_physics_single(input):
    """1. Check that the forward simulated cell-only signal corresponds to an isotropic diffusion tensor (for only one cell)
    'Note': the inverse problem measured diffusivity here will strongly depend on the diffusion time, thus, this test only requires that the cell diffusion be isotropic    
    """

    bvals, bvecs = diffusion_schemes.get_from_default('DBSI_99')
    gtab = gradient_table(bvals, bvecs)
    tenmodel = dti.TensorModel(gtab)
    """
    Make Configuration File
    """
    cwd = os.path.dirname(os.path.abspath(__file__))
    cfg_file = configparser.ConfigParser()
    cfg_file.read(os.path.join(cwd, 'config.ini'))

    cfg_file['SIMULATION']['n_walkers'] = '256000'
    cfg_file['SIMULATION']['DELTA'] = '.1'
    cfg_file['SIMULATION']['dt'] = '.001'
    cfg_file['SIMULATION']['voxel_dims'] = '10'
    cfg_file['SIMULATION']['buffer'] = '0'
    cfg_file['SIMULATION']['void_distance'] = '0'
    cfg_file['SIMULATION']['bvals'] = "'N/A'"
    cfg_file['SIMULATION']['bvecs'] = "'N/A'"
    cfg_file['SIMULATION']['diffusion_scheme'] = "'DBSI_99'"
    cfg_file['SIMULATION']['output_directory'] = "'N/A'"
    cfg_file['SIMULATION']['verbose'] = "'no'"

    cfg_file['FIBERS']['fiber_fractions'] = '0,0'
    cfg_file['FIBERS']['fiber_radii']= '1.0,1.0'
    cfg_file['FIBERS']['thetas'] = '0,0'
    cfg_file['FIBERS']['fiber_diffusions'] = '1.0,2.0'


    cfg_file['CELLS']['cell_fractions'] = '.1'
    cfg_file['CELLS']['cell_radii'] = f'{input}'

    cfg_file['WATER']['water_diffusivity'] = '3.0'

    with open(os.path.join(cwd, 'config.ini'), 'w') as configfile:
        cfg_file.write(configfile)

    """
    Run the test
    """
    cmd = r"simDRIFT"
    cmd += f" simulate --configuration {os.path.join(cwd, 'config.ini')}"
    os.system(cmd)
    signals = sorted(glob.glob(os.getcwd() + os.sep + '*' + os.sep + 'signals' + os.sep + 'cell_signal.nii'), key =os.path.getmtime)
    tenfit = tenmodel.fit(nb.load(signals[-1]).get_fdata())
    assert np.isclose(1e3 * tenfit.ad, 1e3 * tenfit.rd, atol=.1)


def run(save_dir):
   logging.info(f' (20/20) Test Single Cell Physics: verify that the forward simulated cell-only signal, at various cell radii, corresponds to an isotropic diffusion tensor, r = [1.0 um] <-> AD = RD')
   results_dir = os.path.join(save_dir, 'test_signal_shapes_results')
   
   if not os.path.exists(results_dir): os.mkdir(results_dir)
   os.chdir(results_dir)

   cmd = f"pytest {__file__}"
   os.system(cmd)
