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
    def __init__(self, parsed_args_dict) -> None:
        self.args_dict = parsed_args_dict
        pass
    @property
    def n_walkers(self):
        return self.args_dict['n_walkers']
    
    @property
    def fiber_fractions(self):
        return self.args_dict['fiber_fractions']
    
    @property
    def fiber_radii(self):
        return self.args_dict['fiber_radii']
    
    @property
    def thetas(self):
        return self.args_dict['thetas']
    
    @property
    def fiber_diffusions(self):
        return self.args_dict['fiber_diffusions']

    @property
    def cell_fractions(self):
        return self.args_dict['cell_fractions']
    
    @property
    def cell_radii(self):
        return self.args_dict['cell_radii']
    
    @property
    def fiber_configuration(self):
        return self.args_dict['fiber_configuration']
    
    @property
    def water_diffusivity(self):
        return self.args_dict['water_diffusivity']
    
    @property
    def Delta(self):
        return self.args_dict['Delta']
    
    @property
    def dt(self):
        return self.args_dict['dt']
    
    @property
    def voxel_dimensions(self):
        return self.args_dict['voxel_dims']
    
    @property
    def buffer(self):
        return self.args_dict['buffer']
    
    @property 
    def bvecs(self):
        return self.args_dict['input_bvecs']
    
    @property
    def bvals(self):
        return self.args_dict['input_bvals']
    
    @property
    def void_distance(self):
        return self.args_dict['void_dist']

    @property
    def diff_scheme(self):
        return self.args_dict['diff_scheme']
    
    @property
    def verbose(self):
        return self.args_dict['verbose']
    
    @property
    def custom_diff_scheme_flag(self):
        return self.args_dict['CUSTOM_DIFF_SCHEME_FLAG']


class dmri_simulation(Parameters):
    r"""
    Class instance of the forward simulation model of the Pulsed Gradient Spin Echo (PGSE) Experiment.

    Attributes:
        parameters: A dictionary of simulation parameters entered in cli.py.
        fibers: A list of objects.fiber instances
        spins: A list of objects.spin instances
        cells: A list of objects.cell instances
    """

    def __init__(self, args: Dict):
        Parameters.__init__(self, args)
        fibers = []
        spins = []
        cells = []
        return
    
    def run(self,):
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
    path, file = os.path.split(arg)
    simObj = dmri_simulation()
    simObj.from_config(arg)

def run(args):
    if args['verbose'] == 'no':
        with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    simulation_obj = dmri_simulation(args)
                    simulation_obj.run()
    else:
        simulation_obj = dmri_simulation(args)
        simulation_obj.run()
    
