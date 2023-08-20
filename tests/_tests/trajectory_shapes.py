
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


@pytest.mark.parametrize("input, expected", [(100, 100), (256 * 1e3, 256 * 1e3), (1e6,1e6,)])
def test_trajectory_shapes(input, expected):
    """1. Check that the forward simulated trajectories match the number of simulated spins
    """
    """
    Make Configuration File
    """
    cwd = os.path.dirname(os.path.abspath(__file__))
    cfg_file = configparser.ConfigParser()
    cfg_file.read(os.path.join(cwd, 'config.ini'))

    cfg_file['SIMULATION']['n_walkers'] = f'{int(input)}'
    cfg_file['SIMULATION']['DELTA'] = '.001'
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
    trajectories = sorted(glob.glob(os.getcwd() + os.sep + '*' + os.sep + 'trajectories' + os.sep + 'total_trajectories_t1m.npy'), key =os.path.getmtime)    
    assert np.load(trajectories[-1]).shape[0] == expected

def run(save_dir):
    logging.info(f'Test Trajectory Shapes: assert that the forward simulated trajectory matrix matches the size of the input number of spins in the ensemble \n\t  (8/20)-100 spins \n\t  (9/20)-256,000 spins \n\t  (10/20)-1,000,000 spins')
    results_dir = os.path.join(save_dir, 'test_trajectory_shapes_results')
    
    if not os.path.exists(results_dir): os.mkdir(results_dir)
    os.chdir(results_dir)
    cmd = f"pytest {__file__}"
    os.system(cmd)

