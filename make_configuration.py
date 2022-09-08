    
import configparser
from fractions import Fraction
import os 
import numpy as np


fractions = [(.1 , .1), (.2, .2), (.3,.3), (.4, .4), (.5, .5), (.6, .6)]
Thetas = [(0,90), (0,90)]
Diffusions = [(1.0, 2.0)]

for fraction in fractions:
        for theta in Thetas:
                config = configparser.ConfigParser()
                cfgfile = open(r"C:\MCSIM\dMRI-MCSIM-main\run_from_config_test\density_Tests" + os.sep + "simulation_configuration_Theta={}_Fraction={}_Diffusivity{}.ini".format(theta,fraction, Diffusions[0]), 'w')
                config['Simulation Parameters'] = {
                                'numSpins': 100*10**3,
                                'fiberFraction': fraction ,
                                'fiberRadius': 1.0,
                                'Thetas': theta,
                                'fiberDiffusions': (1.0, 2.0),
                                'cellFraction': 0.0,
                                'cellRadii': (3,10),
                                'fiberConfiguration': 'Penetrating',
                                'simulateFibers': True,
                                'simulateCells': False,
                                'simulateExtraEnvironment': True
                                }

                config['Scanning Parameters'] = {
                        'Delta': 10,
                        'dt': 0.001,
                        'voxelDim': 200,
                        'buffer': 50,
                        'path_to_bvals': r'C:\MCSIM\Repo\simulation_data\DBSI\DBSI-99\bval',
                        'path_to_bvecs': r'C:\MCSIM\Repo\simulation_data\DBSI\DBSI-99\bvec'
                        }

                config['Saving Parameters'] = {
                        'path_to_save_file_dir': r"C:\MCSIM\dMRI-MCSIM-main\run_from_config_test\density_Tests"
                }
                config.write(cfgfile)
                cfgfile.close()
