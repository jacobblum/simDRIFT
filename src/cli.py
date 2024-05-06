import sys
import argparse
import src.simulation as simulation
from pathlib import Path
import numba
import os 
import platform

import configparser
import ast

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
        cli_args['n_walkers'] = int(cfg_file['SIMULATION']['n_walkers'])
        #assert args['n_walkers']/(args['voxel_dims']**3) > 1.0," --Simulation requires spin densities > 1.0 per cubic micron"
        
        # Delta
        cli_args['Delta'] = float(cfg_file['SIMULATION']['Delta'])
        assert cli_args['Delta'] > 0, "--Delta must be positive"

        # Delta
        cli_args['delta'] = float(cfg_file['SIMULATION']['delta'])
        assert cli_args['delta'] > 0, "--delta must be positive"
        
        #dt
        cli_args['dt'] = float(cfg_file['SIMULATION']['dt'])
        assert cli_args['dt'] > 0, "--dt must be positive"
        
        # Voxel Dims
        cli_args['voxel_dims'] = float(cfg_file['SIMULATION']['voxel_dims'])
        assert cli_args['voxel_dims'] > 0, "--voxel dims must be positive"
        
        # Void Distance
        cli_args['void_dist'] = float(cfg_file['SIMULATION']['void_distance'])
        assert cli_args['void_dist'] >= 0, "--void_dist must be non-negative"

        cli_args['buffer'] = int(cfg_file['SIMULATION']['buffer'])
        assert cli_args['buffer'] >= 0, "--buffer must be non-negative"

        ## If using custom diffusion scheme... make sure that the bval and bvec paths exist
        if all([ast.literal_eval(cfg_file['SIMULATION']['bvals']) != 'N/A', ast.literal_eval(cfg_file['SIMULATION']['bvecs']) != 'N/A']):
            cli_args['CUSTOM_DIFF_SCHEME_FLAG'] = True

            cli_args['input_bvals'] = ast.literal_eval(cfg_file['SIMULATION']['bvals'])
            cli_args['input_bvecs'] = ast.literal_eval(cfg_file['SIMULATION']['bvecs'])

            assert all([os.path.exists(path) for path in [cli_args['input_bvals'], cli_args['input_bvecs']]])
        else:
            cli_args['CUSTOM_DIFF_SCHEME_FLAG'] = False       

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
        cli_args['fiber_fractions'] = [float(frac) for frac in str(cfg_file['FIBERS']['fiber_fractions']).split(',')]
        for ff in cli_args['fiber_fractions']:
            assert ff >= 0, "--fiber_fractions must be non-negative"
            assert ff < 1, "--fiber_fractions must be less than 1.0"
        assert sum(cli_args['fiber_fractions']) < 1, "--fiber_fractions cannot sum to more than 1.0"

        # Fiber-Radii
        cli_args['fiber_radii'] = [float(rad) for rad in str(cfg_file['FIBERS']['fiber_radii']).split(',')]
        for radius in cli_args['fiber_radii']:
            assert radius > 0, "--fiber_radii must be positive"

        # Thetas
        cli_args['thetas'] = [float(theta) for theta in str(cfg_file['FIBERS']['thetas']).split(',')]
        
        # Fiber Diffusions
        cli_args['fiber_diffusions'] = [float(D0) for D0 in str(cfg_file['FIBERS']['fiber_diffusions']).split(',')]
        for D0 in cli_args['fiber_diffusions']:
            assert D0 >= 0, "--fiber_diffusions must be non-negative"

        #Fiber configuration
        cli_args['fiber_configuration'] = ast.literal_eval(cfg_file['FIBERS']['configuration'])
        assert cli_args['fiber_configuration'] in ['Penetrating', 'penetrating', 'Interwoven', 'interwoven', 'Void', 'void'], 'configuration chosen: {} please choose a fiber configuration from the following: Penetrating, Interwoven, Void'.format(cli_args['fiber_configuration'])
        
        # ------------------------------------------------------------------------------- #
        #                                   Curvature                                     #
        # ------------------------------------------------------------------------------- #

        #kappa
        cli_args['kappa'] = [float(kappa) for kappa in str(cfg_file['CURVATURE']['kappa']).split(',')]
        
        # Attenuation
        cli_args['A'] = [float(A) for A in str(cfg_file['CURVATURE']['Amplitude']).split(',')]

        # Periodicity
        cli_args['P'] = [float(P) for P in str(cfg_file['CURVATURE']['Periodicity']).split(',')]
        for P in cli_args['P']:
            assert P>= 0, "--Periodicity must be greater than  or equal to 0"

        # Enforce self Consistent Fiber Parameters 
        assert all([len(cli_args['fiber_fractions']) == len(param) for param in [cli_args['thetas'], cli_args['fiber_radii'], cli_args['fiber_diffusions']
                                                                                ,cli_args['kappa'], cli_args['A'], cli_args['P']]
                    ]
                    ), "fiber parameters must be consistent with eachother (of equal lengths)"
    
        
        # ------------------------------------------------------------------------------- #
        #                                     Cells                                       #
        # ------------------------------------------------------------------------------- #

        # Cell Fractions
        cli_args['cell_fractions'] = [float(fraction) for fraction in str(cfg_file['CELLS']['cell_fractions']).split(',')]
        for cf in cli_args['cell_fractions']:
            assert cf >= 0, "--cell_fractions must be non-negative"
            assert cf < 1, "--cell_fractions must be less than 1.0"
        assert sum(cli_args['cell_fractions']) < 1, "--cell_fractions cannot sum to more than 1.0"
        
        # Cell Radii 
        cli_args['cell_radii'] = [float(rad) for rad in str(cfg_file['CELLS']['cell_radii']).split(',')]
        for cr in cli_args['cell_radii']:
            assert cr > 0, "--cell_radii must be positive"

        # Enforce self consistent 
        assert len(cli_args['cell_fractions']) == len(cli_args['cell_radii']), "cell parameters must be consistent with eachother (of equal lengths)"

        # ------------------------------------------------------------------------------- #
        #                                     Water                                       #
        # ------------------------------------------------------------------------------- #

        # Water Diffusivity
        cli_args['water_diffusivity'] = float(cfg_file['WATER']['water_diffusivity'])
        assert cli_args['water_diffusivity'] >= 0.0, "--water_diffusivity must be non-negative"

        # Flow Diffusivity
        cli_args['flow_diffusivity'] = float(cfg_file['WATER']['flow_diffusivity'])
        assert cli_args['flow_diffusivity'] >= 0.0, "--flow_diffusivity must be non-negative"
        
        cli_args['verbose'] = ast.literal_eval(str(cfg_file['SIMULATION']['verbose']))
        assert cli_args['verbose'] == 'yes' or cli_args['verbose'] == 'no', "--verbose must be yes or no"

        # ------------------------------------------------------------------------------- #
        #                                     Other                                       #
        # ------------------------------------------------------------------------------- #

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
