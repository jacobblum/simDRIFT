import sys
import argparse
import src.simulation as simulation
from pathlib import Path
import numba
import os 
import platform

import configparser
import ast
import numpy as np

class CLI:
    def __init__(self, subparsers) -> None:
        """Initializes subparsers for input parameters

        :param subparsers: Parsers for each relevant module
        :type subparsers: str
        """   
        self.subparsers = subparsers        
        pass

    def validate_args(self, args):   

        """Validation step for parsed user input arguments

        :param args: Parsed user inputs
        :type args: dictionary
        :return: Parsed and validated arguments
        :rtype: dictionary
        """         
        cli_args = {}
        assert os.path.exists(args['configuration_path']), "Configuration file path cannot be found. Please enter the absolute path to the configuration.ini file"

        cfg_file = configparser.ConfigParser()
        cfg_file.optionxform = str
        cfg_file.read(args['configuration_path'])

        # ------------------------------------------------------------------------------- #
        #                                     Simulation                                  #
        # ------------------------------------------------------------------------------- #

        # N_walkers
        cli_args['n_walkers'] = np.array(int(cfg_file['SIMULATION']['n_walkers'])).astype(np.int32)        
        assert cli_args['n_walkers'] > 0," --n_walkers must be positive"
        
        # Delta
        cli_args['Delta'] = np.array(float(cfg_file['SIMULATION']['Delta'])).astype(np.float32)*1e-3 # sec.
        assert cli_args['Delta'] > 0, "--Delta must be positive"

        # delta
        cli_args['delta'] = np.array(float(cfg_file['SIMULATION']['delta'])).astype(np.float32)*1e-3 # sec.
        assert cli_args['delta'] > 0, "--delta must be positive"
        assert cli_args['Delta'] > cli_args['delta'] 
        
        #dt
        cli_args['dt'] = np.array(float(cfg_file['SIMULATION']['dt'])).astype(np.float32)*1e-3 # sec.
        assert cli_args['dt'] > 0, "--dt must be positive"
        
        #TE 
        cli_args['TE'] = np.array(cli_args['Delta'] + cli_args['delta'] ).astype(np.float32) # sec

        # Voxel Dims
        cli_args['voxel_dimensions'] = np.array(float(cfg_file['SIMULATION']['voxel_dims'])).astype(np.float32)*1e-6 #meters
        assert cli_args['voxel_dimensions'] > 0, "--voxel dims must be positive"
        
        # Void Distance
        cli_args['void_distance'] = np.array(float(cfg_file['SIMULATION']['void_distance'])).astype(np.float32)*1e-6 #meters
        assert cli_args['void_distance'] >= 0, "--void_dist must be non-negative"

        cli_args['buffer'] = np.array(int(cfg_file['SIMULATION']['buffer'])).astype(np.float32) * 1e-6 # meters
        assert cli_args['buffer'] >= 0, "--buffer must be non-negative"

        ## If using custom diffusion scheme... make sure that the bval and bvec paths exist
        if all([ast.literal_eval(cfg_file['SIMULATION']['bvals']) != 'N/A', ast.literal_eval(cfg_file['SIMULATION']['bvecs']) != 'N/A']):
            cli_args['custom_diff_scheme_flag'] = True

            cli_args['bvals'] = ast.literal_eval(cfg_file['SIMULATION']['bvals'])
            cli_args['bvecs'] = ast.literal_eval(cfg_file['SIMULATION']['bvecs'])

            assert all([os.path.exists(path) for path in [cli_args['bvals'], cli_args['bvecs']]])
        else:
            cli_args['custom_diff_scheme_flag'] = False       

        cli_args['diff_scheme'] = ast.literal_eval(cfg_file['SIMULATION']['diffusion_scheme'])
        assert cli_args['diff_scheme'] in ['DBSI_99', 'ABCD', 'NODDI_145']

        cli_args['output_directory'] = ast.literal_eval("{}".format(cfg_file['SIMULATION']['output_directory']))

        if cli_args['output_directory'] != 'N/A':
            assert os.path.exists(cli_args['output_directory'])
        else:
            cli_args['output_directory'] = os.getcwd()

        # ------------------------------------------------------------------------------- #
        #                                     Fibers                                      #
        # ------------------------------------------------------------------------------- #

        # Fiber-Fractions
        cli_args['fiber_fractions'] = np.array([float(frac) for frac in str(cfg_file['FIBERS']['fiber_fractions']).split(',')]).astype(np.float32)
        assert (cli_args['fiber_fractions'] >= 0).all(), "--fiber_fractions must be non-negative"
        assert cli_args['fiber_fractions'].sum() < 1, "--fiber_fractions cannot sum to more than 1.0"
   
        # Fiber-Radii
        cli_args['fiber_radii'] = np.array([float(rad) for rad in str(cfg_file['FIBERS']['fiber_radii']).split(',')]).astype(np.float32)*1e-6 #meters
        assert (cli_args['fiber_radii'] > 0).all(), "--fiber_radii must be positive"

        # Thetas
        cli_args['thetas'] = np.array([float(theta) for theta in str(cfg_file['FIBERS']['thetas']).split(',')]).astype(np.float32)
        
        # Fiber Diffusions
        cli_args['fiber_diffusions'] = np.array([float(D0) for D0 in str(cfg_file['FIBERS']['fiber_diffusions']).split(',')]).astype(np.float32)*1e-9 # m^2 / sec
        assert (cli_args['fiber_diffusions'] > 0).all(), "--fiber_diffusions must be non-negative"

        #Fiber configuration
        cli_args['fiber_configuration'] = ast.literal_eval(cfg_file['FIBERS']['configuration'])
        assert cli_args['fiber_configuration'] in ['Penetrating', 'penetrating', 'Interwoven', 'interwoven', 'Void', 'void'], 'configuration chosen: {} please choose a fiber configuration from the following: Penetrating, Interwoven, Void'.format(cli_args['fiber_configuration'])
        
        # ------------------------------------------------------------------------------- #
        #                                   Curvature                                     #
        # ------------------------------------------------------------------------------- #

        #kappa
        cli_args['kappa'] = np.array([float(kappa) for kappa in str(cfg_file['CURVATURE']['kappa']).split(',')]).astype(np.float32)
        
        # Attenuation
        cli_args['A'] = np.array([float(A) for A in str(cfg_file['CURVATURE']['Amplitude']).split(',')]).astype(np.float32)*1e-6

        # Periodicity
        cli_args['P'] = np.array([float(P) for P in str(cfg_file['CURVATURE']['Periodicity']).split(',')]).astype(np.float32)
        assert (cli_args['P'] >= 0).all(), "--Periodicity must be greater than  or equal to 0"

        # Enforce self Consistent Fiber Parameters 
        assert all([len(cli_args['fiber_fractions']) == len(param) for param in [cli_args['thetas'], cli_args['fiber_radii'], cli_args['fiber_diffusions']
                                                                                ,cli_args['kappa'], cli_args['A'], cli_args['P']]
                    ]
                    ), "fiber parameters must be consistent with eachother (of equal lengths)"
    
        
        # ------------------------------------------------------------------------------- #
        #                                     Cells                                       #
        # ------------------------------------------------------------------------------- #

        # Cell Fractions
        cli_args['cell_fractions'] = np.array([float(fraction) for fraction in str(cfg_file['CELLS']['cell_fractions']).split(',')]).astype(np.float32)
        assert (cli_args['cell_fractions'] >= 0).all(), "--cell_fractions must be non-negative"
        assert cli_args['cell_fractions'].sum() < 1, "--cell_fractions cannot sum to more than 1.0"
        
        # Cell Radii 
        cli_args['cell_radii'] = np.array([float(rad) for rad in str(cfg_file['CELLS']['cell_radii']).split(',')]).astype(np.float32)*1e-6 #meters
        assert (cli_args['cell_radii'] > 0).all(), "--cell_radii must be positive"

        # Enforce self consistent 
        assert len(cli_args['cell_fractions']) == len(cli_args['cell_radii']), "cell parameters must be consistent with eachother (of equal lengths)"

        # ------------------------------------------------------------------------------- #
        #                                     Water                                       #
        # ------------------------------------------------------------------------------- #

        # Water Diffusivity
        cli_args['water_diffusivity'] = np.array(float(cfg_file['WATER']['water_diffusivity'])).astype(np.float32)*1e-9 # m^2 / sec.
        assert cli_args['water_diffusivity'] >= 0.0, "--water_diffusivity must be non-negative"

        # ------------------------------------------------------------------------------- #
        #                                     Other                                       #
        # ------------------------------------------------------------------------------- #

        
        cli_args['verbose'] = ast.literal_eval(str(cfg_file['SIMULATION']['verbose']))
        assert cli_args['verbose'] == 'yes' or cli_args['verbose'] == 'no', "--verbose must be yes or no"
        cli_args['verbose'] = (cli_args['verbose'] == 'yes') 
       
        cli_args['draw_voxel'] = ast.literal_eval(str(cfg_file['SIMULATION']['draw_voxel']))
        assert cli_args['draw_voxel'] == 'yes' or cli_args['draw_voxel'] == 'no', "--draw_voxel must be yes or no"
        cli_args['draw_voxel'] = (cli_args['draw_voxel'] == 'yes') 
       
        
        ## Check that GPU is available 
        
        if platform.system() != 'Darwin':
            assert numba.cuda.is_available(), "Trying to use Cuda device, " \
                                            "but Cuda device is not available."

        elif platform.system() == 'Darwin':
            print('Mac OS detected ... please note that the simulation Geometry will be instantiated; however, the random walk will not be performed!' \
                ' This functionality mainly exists for DEBUGGING!') 
        
        cli_args['cfg_path'] = args['configuration_path']

        return cli_args
    
    def run(self, args):
        """Run simulation using parsed user inputs

        :param args: User inputs for relevant parameters
        :type args: dictionary
        """
        simulation.run(args)


    def add_subparser_args(self) -> argparse:
        """Defines subparsers for each simulation parameter.

        :return: argparse object containing subparsers for each simulation parameter
        :rtype: argparse
        """  
        
        subparser = self.subparsers.add_parser("simulate",
                                        description="Simulate PGSE experiment"
                                        "on custom defined biological domain",
                                        )

        """  Simulation Parameters """

        subparser.add_argument("--configuration", nargs = None, type = str, dest = 'configuration_path', required = True, 
                               help = "please enter the absolute path to the simulation configuration file")


        return self.subparsers
