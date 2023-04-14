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

        self.numCells = set_voxel_configuration._set_num_cells(self.parameters['cell_fractions'],
                                                               self.parameters['cell_radii'],
                                                               self.parameters['voxel_dims'],
                                                               self.parameters['buffer']
                                                               )

        self.fibers = set_voxel_configuration._place_fiber_grid(self.parameters['fiber_fractions'],
                                                                self.parameters['fiber_radii'],
                                                                self.parameters['fiber_diffusions'],
                                                                self.parameters['thetas'],
                                                                self.parameters['voxel_dims'],
                                                                self.parameters['buffer'],
                                                                self.parameters['void_dist'],
                                                                self.parameters['fiber_configuration']
                                                                )

        self.cells = set_voxel_configuration._place_cells(self.parameters['cell_fractions'],
                                                          self.parameters['cell_radii'],
                                                          self.fibers,
                                                          self.parameters['fiber_configuration'],
                                                          self.parameters['voxel_dims'],
                                                          self.parameters['buffer'],
                                                          self.parameters['void_dist']
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
        try:
            self.set_parameters(args)
            self.set_voxel()
            diffusion._simulate_diffusion(self,
                                        self.spins,
                                        self.cells,
                                        self.fibers,
                                        self.parameters['Delta'],
                                        self.parameters['dt'],
                                        )
            
            save._save_data(self,
                        self.path_to_save,
                        plot_xyz=False)
        

        except KeyboardInterrupt:
            sys.stdout.write('Keyboard interupt. Terminated without saving. \n')

    
    
    
   

    def spins_in_voxel(self, trajectoryT1m, trajectoryT2p):
        """
         Helper function to ensure that the spins at time T2p are wtihin the self.voxelDims x self.voxelDims x inf imaging voxel

        Parameters
        ----------
        trajectoryT1m: N_{spins} x 3 ndarray
            The initial spin position at time t1m

        trajectoryT2p: N_{spins} x 3 ndarray
            The spin position at time t2p

        Returns
        -------
        traj1_vox: (N, 3) ndarray
            Position at T1m of the spins which stay within the voxel
        traj2_vox: (N, 3) ndarray
            Position at T2p of the spins which stay within the voxel

        Notes
        -----
        None

        References
        ----------
        None

        """

        traj1_vox = []
        traj2_vox = []

        for i in range(trajectoryT1m.shape[0]):
            if np.amin(trajectoryT2p[i, 0:2]) >= 0 + 0.5*self.buffer and np.amax(trajectoryT2p[i, 0:2]) <= self.voxelDims + 0.5*self.buffer:
                traj1_vox.append(trajectoryT1m[i, :])
                traj2_vox.append(trajectoryT2p[i, :])
        return np.array(traj1_vox), np.array(traj2_vox)


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
