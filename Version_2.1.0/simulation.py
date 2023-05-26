import numpy as np
import numba
from numba import jit, njit, cuda, int32, float32
import os
import glob as glob
import configparser
import contextlib
from setup import spin_init_positions, set_voxel_configuration
import sys
import diffusion
import save
from jp import linalg
import logging

class dmri_simulation:
    def __init__(self):

        spin_positions_t1m = 0.0
        fiber1PositionsT1m = 0.0
        fiber1PositionsT2p = 0.0
        fiber2PositionsT1m = 0.0
        fiber2PositionsT2p = 0.0
        cellPositionsT1m = 0.0
        cellPositionsT2p = 0.0
        extraPositionsT1m = 0.0
        extraPositionsT2p = 0.0
        spinInFiber1_i = 0.0
        spinInFiber2_i = 0.0
        spinInCell_i = 0.0
        fiberRotationReference = 0.0
        rotMat = 0.0
        path_to_save = ''
        random_state = 0.0
        parameters = {}
        fibers = None

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
                                                          self.fibers
                                                          )
        
        spin_init_positions._find_spin_locations(self, 
                                                 self.spins, 
                                                 self.cells, 
                                                 self.fibers)
        
        return

    def run(self, args):
        log_file = os.path.join(os.getcwd() + os.sep + 'log')
        logging.basicConfig(level = logging.INFO,
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
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        try:
            numba.cuda.detect()
        except:
            raise Exception(
                "Numba was unable to detect a CUDA GPU. To run the simulation,"
                + " check that the requirements are met and CUDA installation"
                + " path is correctly set up: "
                + "https://numba.pydata.org/numba-doc/dev/cuda/overview.html"
            )
    simulation_obj = dmri_simulation()
    simulation_obj.run(args)
