import numpy as np
import numba
from numba import jit, njit, cuda, int32, float32
import os
import glob as glob
import contextlib
from datetime import datetime
import sys
import src.physics.diffusion as diffusion
import src.save as save 
from src.setup import set_voxel_configuration
import logging
from typing import Dict
from . import gradients 


class dmri_simulation():
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
        """Initializes dmri_simulation class.

        :param args: Parsed input arguments
        :type args: Dict
        """        
        self.__dict__.update(args)
        self.fibers = []
        self.spins  = []
        self.cells  = []
        self.G = gradients.pgse(self)
        return
    
    def run(self,):
        """Runs simDRIFT with user inputs
        """      
  
        verbose = {True: logging.INFO, False: logging.WARNING}
        log_file = os.path.join(os.getcwd(),'log')
        
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
            
            self.results_directory = os.path.join(self.output_directory, f"{datetime.now().strftime('%Y%m%d_%H%M')}_simDRIFT_Results")
            if not os.path.exists(self.results_directory): os.mkdir(self.results_directory)

            set_voxel_configuration.setup(self)
            diffusion._simulate_diffusion(self)
            save._save_data(self)
        
        except KeyboardInterrupt:
            sys.stdout.write('\n')
            logging.info('Keyboard interrupt. Terminated without saving.')
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
    
