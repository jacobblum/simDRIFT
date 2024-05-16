
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

@pytest.mark.parametrize("expected", [('.npy'),])
def test_trajectory_types(expected):
    """1. Check that the forward simulated trajectory is a .npy file
    """

    """
    Make Configuration File
    """
    
    cwd = os.path.dirname(os.path.abspath(__file__))
    cfg_file = configparser.ConfigParser()
    cfg_file.optionxform = str
    cfg_file.read(os.path.join(cwd, 'config.ini'))

    cfg_file['SIMULATION']['n_walkers'] = '256000'
    cfg_file['SIMULATION']['Delta'] = '0.20'
    cfg_file['SIMULATION']['delta'] = '.010'
    cfg_file['SIMULATION']['dt'] = '.001'
    cfg_file['SIMULATION']['voxel_dims'] = '10'
    cfg_file['SIMULATION']['buffer'] = '0'
    cfg_file['SIMULATION']['void_distance'] = '0'
    cfg_file['SIMULATION']['bvals'] = "'N/A'"
    cfg_file['SIMULATION']['bvecs'] = "'N/A'"
    cfg_file['SIMULATION']['diffusion_scheme'] = "'DBSI_99'"
    cfg_file['SIMULATION']['output_directory'] = "'N/A'"
    cfg_file['SIMULATION']['verbose'] = "'no'"
    cfg_file['SIMULATION']['draw_voxel'] = "'no'"

    cfg_file['FIBERS']['fiber_fractions'] = '0,0'
    cfg_file['FIBERS']['fiber_radii']= '1.0,1.0'
    cfg_file['FIBERS']['thetas'] = '0,0'
    cfg_file['FIBERS']['fiber_diffusions'] = '1.0,2.0'
    cfg_file['FIBERS']['configuration'] = "'Penetrating'"

    cfg_file['CURVATURE']['kappa'] = '1.0,1.0'
    cfg_file['CURVATURE']['Amplitude'] = '0.0,0.0'
    cfg_file['CURVATURE']['Periodicity'] = '1.0,1.0'
    
    cfg_file['CELLS']['cell_fractions'] = '0,0'
    cfg_file['CELLS']['cell_radii'] = '1.0,1.0'

    cfg_file['WATER']['water_diffusivity'] = '3.0'

    with open(os.path.join(cwd, 'config.ini'), 'w') as configfile:
        cfg_file.write(configfile)


    """
    Run the test
    """
    cmd =  f"python "
    cmd += f"{os.path.join( Path(__file__).parents[2], 'master_cli.py')}"
    cmd += f" simulate --configuration {os.path.join(cwd, 'config.ini')}"

    os.system(cmd)
    
    trajectories = glob.glob(os.getcwd() + os.sep + '*' + os.sep + 'trajectories' + os.sep + '*')
    assert all([os.path.splitext(trajectory)[-1:] == (expected,) for trajectory in trajectories])

def run(save_dir):
    logging.info(f' (2/20) Test Trajectory File Types: assert that the forward trajectory matrix is a .npy file')
    results_dir = os.path.join(save_dir, 'test_trajectory_types_results')
    if not os.path.exists(results_dir): os.mkdir(results_dir)
    os.chdir(results_dir)
    cmd = f"pytest {__file__}"
    os.system(cmd)