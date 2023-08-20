
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


@pytest.mark.parametrize("input, expected", [('DBSI_99', 99), ('ABCD', 103), ('NODDI_145', 145)])
def test_signal_shapes(input, expected):
    """1. Check that the forward simulated signal matches the number of bvals and bvecs used in the 'imaging' experiment 
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
    cfg_file['SIMULATION']['bvals'] = "'N/A'"
    cfg_file['SIMULATION']['bvecs'] = "'N/A'"
    cfg_file['SIMULATION']['diffusion_scheme'] = f"'{input}'"
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


    """
    Run the test
    """
    cmd = r"simDRIFT"
    cmd += f" simulate --configuration {os.path.join(cwd, 'config.ini')}"
    os.system(cmd)
    signals = sorted(glob.glob(os.getcwd() + os.sep + '*' + os.sep + 'signals' + os.sep + 'total_signal.nii'), key =os.path.getmtime)
    assert nb.load(signals[-1]).get_fdata().shape == (expected,)

def run(save_dir):
    logging.info(f'Test Signal Shapes: assert that the forward simulated signal shapes correspond to the input diffusion schemes \n\t  (4/20)-DBSI-99-Direction \n\t  (5/20)-ABCD-103-Direction \n\t  (6/20)-NODDI-145-Direction')
    results_dir = os.path.join(save_dir, 'test_signal_shapes_results')
    
    if not os.path.exists(results_dir): os.mkdir(results_dir)
    os.chdir(results_dir)

    cmd = f"pytest {__file__}"
    os.system(cmd)

