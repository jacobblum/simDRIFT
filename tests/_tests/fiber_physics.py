
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



@pytest.mark.parametrize("input, expected", [((1.0,2.0,2.0), (1.0,2.0,2.0)), ((1.0,1.5,2.0), (1.0,1.5,2.0)), ((1.0,1.0,1.5), (1.0,1.0,1.5))])
def test_fiber_physics_multi(input, expected):
    """1. Check that the forward simulated fiber-only signal corresponds to a diffusion tensor matching the input fiber diffusivities (with multiple fibers)
    2. Check that the forward simulated fiber-only signal corresponds to an anisotropic diffusion tensor (with multiple fibers)
    """
    bvals, bvecs = diffusion_schemes.get_from_default('DBSI_99')
    gtab = gradient_table(bvals, bvecs)
    tenmodel = dti.TensorModel(gtab)
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
    cfg_file['SIMULATION']['DELTA'] = '1'
    cfg_file['SIMULATION']['dt'] = '.001'
    cfg_file['SIMULATION']['voxel_dims'] = '20'
    cfg_file['SIMULATION']['buffer'] = '0'
    cfg_file['SIMULATION']['void_distance'] = '0'
    cfg_file['SIMULATION']['bvals'] = "'N/A'"
    cfg_file['SIMULATION']['bvecs'] = "'N/A'"
    cfg_file['SIMULATION']['diffusion_scheme'] = "'DBSI_99'"
    cfg_file['SIMULATION']['output_directory'] = "'N/A'"
    cfg_file['SIMULATION']['verbose'] = "'no'"

    cfg_file['FIBERS']['fiber_fractions'] = '.3,.3,.3'
    cfg_file['FIBERS']['fiber_radii']= '1.0,1.0,1.0'
    cfg_file['FIBERS']['thetas'] = '0,0,0'
    cfg_file['FIBERS']['fiber_diffusions'] = f'{input[0]},{input[1]},{input[2]}'

    cfg_file['CELLS']['cell_fractions'] = '0.,0.'
    cfg_file['CELLS']['cell_radii'] = '1.0,1.0'

    cfg_file['WATER']['water_diffusivity'] = '3.0'

    with open(os.path.join(cwd, 'config.ini'), 'w') as configfile:
        cfg_file.write(configfile)


    """
    Run the test
    """

    cmd = r"simDRIFT"
    cmd += f" simulate --configuration {os.path.join(cwd, 'config.ini')}"
    os.system(cmd)
    fiber_1_signals = sorted(glob.glob(os.getcwd() + os.sep + '*' + os.sep + 'signals' + os.sep + 'fiber_1_signal.nii'), key =os.path.getmtime)
    fiber_2_signals = sorted(glob.glob(os.getcwd() + os.sep + '*' + os.sep + 'signals' + os.sep + 'fiber_2_signal.nii'), key =os.path.getmtime)
    fiber_3_signals = sorted(glob.glob(os.getcwd() + os.sep + '*' + os.sep + 'signals' + os.sep + 'fiber_3_signal.nii'), key =os.path.getmtime)

    fiber_1_signal = nb.load(fiber_1_signals[-1]).get_fdata()
    fiber_2_signal = nb.load(fiber_2_signals[-1]).get_fdata()
    fiber_3_signal = nb.load(fiber_3_signals[-1]).get_fdata()   

    tenfit_1 = tenmodel.fit(fiber_1_signal)
    tenfit_2 = tenmodel.fit(fiber_2_signal)
    tenfit_3 = tenmodel.fit(fiber_3_signal)

    assert (np.isclose(1e3 * np.array([tenfit_1.ad, tenfit_2.ad, tenfit_3.ad]), np.array(expected), atol = 0.1).all())
    assert (np.array([tenfit_1.fa, tenfit_2.fa, tenfit_3.fa]) > .20).all()



def run(save_dir):
    logging.info(f'Test Fiber Physics: verify that the forward simulated respective fiber-only signals corresponds to a diffusion tensor matching the input fiber diffusivity parameter, and verify that this diffusion tensor is anisotropic \n\t  (14/20)- [D_fiber_1, D_fiber_2, D_fiber_3] = [1.0, 2.0, 2.0] (um^2 /ms) <-> AD >> RD \
                    \n\t  (15/20)- [D_fiber_1, D_fiber_2, D_fiber_3] = [1.0, 1.5, 2.0] (um^2 /ms) <-> AD >> RD\
                    \n\t  (16/20)- [D_fiber_1, D_fiber_2, D_fiber_3] = [1.0, 1.0, 1.5] (um^2 /ms) <-> AD >> RD')
    results_dir = os.path.join(save_dir, 'test_fiber_physics_results')
   
    if not os.path.exists(results_dir): os.mkdir(results_dir)
    os.chdir(results_dir)

    cmd = f"pytest {__file__}"
    os.system(cmd)