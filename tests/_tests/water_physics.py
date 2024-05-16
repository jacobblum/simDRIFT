
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

@pytest.mark.parametrize("input, expected", [(3.0, 3.0), (1.0, 1.0), (2.0, 2.0)])
def test_water_physics(input, expected):
   """1. Check that the forward simulated water-only signal corresponds to a diffusion tensor matching the input water diffusivity
      2. Check that the forward simulated water-only signal corresponds to an isotropic diffusion tensor 
   """

   bvals, bvecs = diffusion_schemes.get_from_default('DBSI_99')
   gtab = gradient_table(bvals, bvecs)
   tenmodel = dti.TensorModel(gtab)
   """
   Make Configuration File
   """
   cwd = os.path.dirname(os.path.abspath(__file__))
   cfg_file = configparser.ConfigParser()
   cfg_file.optionxform = str
   cfg_file.read(os.path.join(cwd, 'config.ini'))

   cfg_file['SIMULATION']['n_walkers'] = '256000'
   cfg_file['SIMULATION']['Delta'] = '1.0'
   cfg_file['SIMULATION']['delta'] = '.10'
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

   cfg_file['CELLS']['cell_fractions'] = '.0,.0'
   cfg_file['CELLS']['cell_radii'] = '1.0,1.0'

   cfg_file['WATER']['water_diffusivity'] = f'{input}'

   with open(os.path.join(cwd, 'config.ini'), 'w') as configfile:
      cfg_file.write(configfile)

   """
   Run the test
   """
   cmd =  f"python "
   cmd += f"{os.path.join( Path(__file__).parents[2], 'master_cli.py')}"
   cmd += f" simulate --configuration {os.path.join(cwd, 'config.ini')}"

   os.system(cmd)

   signals = sorted(glob.glob(os.getcwd() + os.sep + '*' + os.sep + 'signals' + os.sep + 'water_signal.nii'), key =os.path.getmtime)
   tenfit = tenmodel.fit(nb.load(signals[-1]).get_fdata())
   assert np.isclose(1e9 * tenfit.ad, expected, atol = .1) 
   assert np.isclose(1e9 * tenfit.ad, 1e9 * tenfit.rd, atol=.1)



def run(save_dir):
   logging.info(f'Test Water Physics: verify that the forward simulated water-only signal corresponds to a diffusion tensor matching the input water diffusivity parameter, and verify that this diffusion tensor is isotropic \n\t  (11/20)- D_water = 3.0 um^2 / ms <-> AD = RD = 3.0 um^2 / ms  \n\t  (12/20)- D_water = 2.0 um^2 / ms <-> AD = RD = 2.0 um^2 / ms \n\t  (13/20)- D_water = 1.0 um^2 / ms <-> AD = RD = 1.0 um^2 / ms')
   results_dir = os.path.join(save_dir, 'test_signal_shapes_results')
   
   if not os.path.exists(results_dir): os.mkdir(results_dir)
   os.chdir(results_dir)

   cmd = f"pytest {__file__}"
   os.system(cmd)
