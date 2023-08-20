
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


@pytest.mark.parametrize("input, expected", [((os.path.join(Path(__file__).parents[2], 'src' + os.sep + 'data' + os.sep + 'bval99'), os.path.join(Path(__file__).parents[2], 'src' + os.sep + 'data' + os.sep + 'bvec99')), 99)])
def test_custom_diffusion_scheme(input, expected):
    """1. Check that the forward simulated signal matches the number of bvals and bvecs used in the PGSE experiment w/ 'custom' 
       bvals and bvecs (i.e., loaded in from a path)
    """
    """
    Make Configuration File
    """
    cwd = os.path.dirname(os.path.abspath(__file__))
    cfg_file = configparser.ConfigParser()
    cfg_file.read(os.path.join(cwd, 'config.ini'))

    cfg_file['SIMULATION']['n_walkers'] = '256000'
    cfg_file['SIMULATION']['DELTA'] = '.001'
    cfg_file['SIMULATION']['dt'] = '.001'
    cfg_file['SIMULATION']['voxel_dims'] = '10'
    cfg_file['SIMULATION']['buffer'] = '0'
    cfg_file['SIMULATION']['void_distance'] = '0'
    cfg_file['SIMULATION']['bvals'] = f"r'{input[0]}'"
    cfg_file['SIMULATION']['bvecs'] = f"r'{input[1]}'"
    cfg_file['SIMULATION']['diffusion_scheme'] = "'DBSI_99'"
    cfg_file['SIMULATION']['output_directory'] = "'N/A'"
    cfg_file['SIMULATION']['verbose'] = "'no'"


    cfg_file['FIBERS']['fiber_fractions'] = '0,0'
    cfg_file['FIBERS']['fiber_radii']= '1.0,1.0'
    cfg_file['FIBERS']['thetas'] = '0,0'
    cfg_file['FIBERS']['fiber_diffusions'] = '1.0,2.0'
    
    
    cfg_file['CELLS']['cell_fractions'] = '0,0'
    cfg_file['CELLS']['cell_radii'] = '1.0,1.0'

    cfg_file['WATER']['water_diffusivity'] = '3.0'

    with open(os.path.join(cwd, 'config.ini'), 'w') as configfile:
        cfg_file.write(configfile)

    cmd = r"simDRIFT"
    cmd += f" simulate --configuration {os.path.join(cwd, 'config.ini')}"
    os.system(cmd)
    signal = nb.load(glob.glob(os.getcwd() + os.sep + '*' + os.sep + 'signals' + os.sep + 'water_signal.nii')[0]).get_fdata()  
    assert signal.shape == (expected,)



def run(save_dir):
   logging.info(f' (7/20) Test Custom bval/bvec files: assert that the forward simulated signal induced by a custom, input-supplied bvec and bval file matches the shapes from the specified custom diffusion scheme')
   results_dir = os.path.join(save_dir, 'test_custom_d_scheme_results')
   if not os.path.exists(results_dir): os.mkdir(results_dir)
   os.chdir(results_dir)
   cmd = f"pytest {__file__}"
   os.system(cmd)