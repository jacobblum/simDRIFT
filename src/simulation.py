import numpy as np
import numba
from numba import jit, njit, cuda, int32, float32
import os
import glob as glob
import contextlib
import sys
import src.physics.diffusion as diffusion
import src.save as save 
from src.setup import set_voxel_configuration
import logging
from typing import Dict

class Parameters:
    """Class for simulation parameters and user inputs.
    """    
    def __init__(self, parsed_args_dict) -> None:
        """Initializes parameter fields.

        :param parsed_args_dict: Dictionary of parsed arguments
        :type parsed_args_dict: dictionary
        """        
        self.args_dict = parsed_args_dict
        pass
    @property
    def n_walkers(self):
        """Class property containing the number of spins to simulate

        :return: Dictionary entry for ``n_walkers``
        """        
        return self.args_dict['n_walkers']
    
    @property
    def fiber_fractions(self):
        """Class property containing the fiber density of each bundle

        :return: Dictionary entry for ``fiber_fractions``
        """  
        return self.args_dict['fiber_fractions']
    
    @property
    def fiber_radii(self):
        """Class property containing the radius of each fiber, for each bundle

        :return: Dictionary entry for ``fiber_radii``
        """  
        return self.args_dict['fiber_radii']
    
    @property
    def thetas(self):
        """Class property containing the orientation angles (w.r.t. the `y`-axis) for each fiber bundle

        :return: Dictionary entry for ``thetas``
        """  
        return self.args_dict['thetas']
    
    @property
    def fiber_diffusions(self):
        """Class property containing the intrinsic diffusivities for each fiber bundle

        :return: Dictionary entry for ``fiber_diffusions``
        """  
        return self.args_dict['fiber_diffusions']

    @property
    def cell_fractions(self):
        """Class property containing the volume fraction for each cell type

        :return: Dictionary entry for ``cell_fractions``
        """  
        return self.args_dict['cell_fractions']
    
    @property
    def cell_radii(self):
        """Class property containing the radius of each cell type

        :return: Dictionary entry for ``cell_radii``
        """  
        return self.args_dict['cell_radii']
    
    @property
    def fiber_configuration(self):
        """Class property containing the fiber geometry class to be simulated

        :return: Dictionary entry for ``fiber_configuration``
        """  
        return self.args_dict['fiber_configuration']
    
    @property
    def water_diffusivity(self):
        """Class property containing the diffusivity of free water

        :return: Dictionary entry for ``water_diffusivity``
        """  
        return self.args_dict['water_diffusivity']
    
    @property
    def Delta(self):
        """Class property containing the diffusion time to simulate

        :return: Dictionary entry for ``Delta``
        """  
        return self.args_dict['Delta']
    
    @property
    def dt(self):
        """Class property containing the desired time-step (and pulse duration, under the narrow pulse approximation)

        :return: Dictionary entry for ``dt``
        """  
        return self.args_dict['dt']
    
    @property
    def voxel_dimensions(self):
        """Class property containing the side length of the isotropic voxel to be simulated

        :return: Dictionary entry for ``voxel_dims``
        """  
        return self.args_dict['voxel_dims']
    
    @property
    def buffer(self):
        """Class property containing the desired buffer length to be added to the voxel dimensions

        :return: Dictionary entry for ``buffer``
        """  
        return self.args_dict['buffer']
    
    @property 
    def bvecs(self):
        """Class property containing the path to the file containing the b-vectors (if using a custom scheme)

        :return: Dictionary entry for ``input_bvecs``
        """  
        return self.args_dict['input_bvecs']
    
    @property
    def bvals(self):
        """Class property containing the path to the file containing the b-values (if using a custom scheme)

        :return: Dictionary entry for ``input_bvals``
        """  
        return self.args_dict['input_bvals']
    
    @property
    def void_distance(self):
        """Class property containing the void distance (if ``fiber_configuration`` = ``Void``)

        :return: Dictionary entry for ``void_dist``
        """  
        return self.args_dict['void_dist']

    @property
    def diff_scheme(self):
        """Class property containing the selected included diffusion scheme

        :return: Dictionary entry for ``diff_scheme``
        """  
        return self.args_dict['diff_scheme']
    
    @property
    def verbose(self):
        """Class property containing the flag for turning terminal outputs on and off

        :return: Dictionary entry for ``verbose``
        """  
        return self.args_dict['verbose']
    
    @property
    def custom_diff_scheme_flag(self):
        """Class property containing the flag for using a custom diffusion scheme

        :return: Dictionary entry for ``CUSTOM_DIFF_SCHEME_FLAG``
        """  
        return self.args_dict['CUSTOM_DIFF_SCHEME_FLAG']


class dmri_simulation(Parameters):
    """ Class instance of the forward simulation model of the Pulsed Gradient Spin Echo (PGSE) Experiment.

    :param Parameters: A dictionary of simulation parameters entered in cli.py.
    :type Parameters: dictionary
    :param fibers: A list of objects.fiber instances
    :type fibers: list
    :param spins: A list of objects.spin instances
    :type spins: list
    :param cells: A list of objects.cell instances
    :type cells: list
    """ 

    def __init__(self, args: Dict):
        """Initializes properties from the `Parameters`_ class.

        :param args: Parsed input arguments
        :type args: Dict
        """        
        Parameters.__init__(self, args)
        fibers = []
        spins = []
        cells = []
        return
    
    def run(self,):
        """Runs simDRIFT with user inputs
        """        
        verbose = {'yes': logging.INFO,'no': logging.WARNING}
        log_file = os.path.join(os.getcwd() + os.sep + 'log')
        
        logging.getLogger('numba').setLevel(logging.WARNING)
        logging.getLogger('numpy').setLevel(logging.WARNING)
        logging.basicConfig(level = verbose[self.verbose],
                            format = 'simDRIFT: %(message)s',
                            filename = log_file,
                            filemode = 'w')
        console = logging.StreamHandler()
        formatter = logging.Formatter("simDRIFT: %(message)s")
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
        logging.info('Running simDRIFT')

        try:
            set_voxel_configuration.setup(self)
            diffusion._simulate_diffusion(self)
            save._save_data(self)
        
        except KeyboardInterrupt:
            sys.stdout.write('\n')
            logging.info('Keyboard interupt. Terminated without saving.')
            exit()
 
def dmri_sim_wrapper(arg):
    """Wrapper to launch the simulator

    :param arg: Parsed arguments
    :type arg: str
    """    
    path, file = os.path.split(arg)
    simObj = dmri_simulation()
    simObj.from_config(arg)

def run(args):
    """Function to invoke the simulator

    :param arg: Parsed arguments
    :type arg: str
    """    
    if args['verbose'] == 'no':
        with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    simulation_obj = dmri_simulation(args)
                    simulation_obj.run()
    else:
        simulation_obj = dmri_simulation(args)
        simulation_obj.run()
    
