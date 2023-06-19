import numpy as np
import numba
from numba import jit, njit, cuda, int32, float32
import os
import glob as glob
import contextlib
from setup import spin_init_positions, set_voxel_configuration
import sys
import diffusion
import save
import logging

class dmri_simulation:
    r"""
    Class instance of the forward simulation model of the Pulsed Gradient Spin Echo (PGSE) Experiment.

    Attributes:
        parameters: A dictionary of simulation parameters entered in cli.py.
        fibers: A list of objects.fiber instances
        spins: A list of objects.spin instances
        cells: A list of objects.cell instances
    """

    def __init__(self):
        parameters = {}
        fibers = []
        spins = []
        cells = []
        return

    def set_parameters(self, args):
        self.parameters = args
        return

    def set_voxel(self):

        self.fibers = set_voxel_configuration._place_fiber_grid(self.parameters['fiber_fractions'],
                                                    self.parameters['fiber_radii'],
                                                    self.parameters['fiber_diffusions'],
                                                    self.parameters['thetas'],
                                                    self.parameters['voxel_dims'],
                                                    self.parameters['buffer'],
                                                    self.parameters['void_dist'],
                                                    self.parameters['fiber_configuration']
                                                    )
    
        self.cells = set_voxel_configuration._place_cells(self.fibers,
                                                        self.parameters['cell_radii'],
                                                        self.parameters['cell_fractions'],
                                                        self.parameters['fiber_configuration'],
                                                        self.parameters['voxel_dims'],
                                                        self.parameters['buffer'],
                                                        self.parameters['void_dist'],
                                                        self.parameters['water_diffusivity']
                                                        )
    
    
        self.spins = set_voxel_configuration._place_spins(self.parameters['n_walkers'],
                                                        self.parameters['voxel_dims'],
                                                        self.fibers)
      
        spin_init_positions._find_spin_locations(self, 
                                                 self.spins, 
                                                 self.cells, 
                                                 self.fibers)
        
        return

    def run(self, args):
        verbose = {'yes': logging.INFO,'no': logging.WARNING}
        log_file = os.path.join(os.getcwd() + os.sep + 'log')
        logging.basicConfig(level = verbose[args['verbose']],
                            format = 'dMRI-SIM: %(message)s',
                            filename = log_file,
                            filemode = 'w')
        console = logging.StreamHandler()
        formatter = logging.Formatter("dMRI-SIM: %(message)s")
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
        logging.info('Running dMRI-SIM')

        try:
            self.set_parameters(args)
            self.set_voxel()
            diffusion._simulate_diffusion(self,
                                          self.spins,
                                          self.cells,
                                          self.fibers,
                                          self.parameters['Delta'],
                                          self.parameters['dt'],
                                          self.parameters['water_diffusivity']
                                        )
            
            save._save_data(self,
                            self.spins,
                            self.parameters['Delta'],
                            self.parameters['dt'],
                            'DBSI_99')
        
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
                    simulation_obj = dmri_simulation()
                    simulation_obj.run(args)
    else:
        simulation_obj = dmri_simulation()
        simulation_obj.run(args)
