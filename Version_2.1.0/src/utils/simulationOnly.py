import contextlib
from email.policy import default
import multiprocessing as mp
from multiprocessing.sharedctypes import Value
import numpy as np 
import numba 
from numba import jit, cuda, int32, float32
from numba.cuda import random 
from numba.cuda.random import xoroshiro128p_normal_float32,  create_xoroshiro128p_states
import math
import matplotlib.pyplot as plt
import time 
import os  
from tqdm import tqdm
import nibabel as nb
import glob as glob 
import configparser
from ast import literal_eval
from multiprocessing import Process
import shutil
import jp as jp
import walk_in_fiber
import walk_in_cell
import walk_in_extra_environ
import spin_init_positions
import sys
import diffusion
import set_voxel_configuration
import save_simulated_data


class dmri_simulation:
    def __init__(self):
        numSpins =                  0.0 
        numFibers =                 0.0
        fiberFraction =             0.0
        fiberRadius =               0.0 #um
        Delta =                     0.0 #ms
        dt =                        0.0 #ms
        delta =                     0.0 #ms
        numCells =                  0.0
        cellFraction =              0.0
        cellRadii =                 0.0 #um
        spinPositionsT1m =          0.0
        fiber1PositionsT1m =         0.0
        fiber1PositionsT2p =         0.0
        fiber2PositionsT1m =         0.0
        fiber2PositionsT2p =         0.0
        cellPositionsT1m =          0.0
        cellPositionsT2p =          0.0
        extraPositionsT1m =         0.0
        extraPositionsT2p =         0.0
        spinInFiber1_i =            0.0
        spinInFiber2_i =            0.0
        spinInCell_i =              0.0
        fiberRotationReference =    0.0
        Thetas =                    0.0
        fiberDiffusions  =          0.0
        rotMat =                    0.0
        fiberConfiguration =        ''
        voidDist =                  0.0
        buffer =                    0.0
        bvals =                     0.0
        bvecs =                     0.0
        cfg_path =                  ''
        path_to_save =              ''
        random_state =              0.0

        return
    
    def set_parameters(self,**kwargs):
        sys.stdout.write('\nChecking input validity...')

        num_spins = kwargs.pop('num_spins',                     None)
        fiber_fraction = kwargs.pop('fiber_fraction',           None)
        fiber_radii = kwargs.pop('fiber_radius',                None)
        thetas = kwargs.pop('thetas',                           None)
        fiber_diffusions = kwargs.pop('fiber_diffusions',       None)
        cell_fraction = float(kwargs.pop('cell_fraction',       None))
        cell_radii = kwargs.pop('cell_radii',                   None)
        fiber_configuration = kwargs.pop('fiber_configuration', None)
        Delta = float(kwargs.pop('Delta',                       None))
        dt = kwargs.pop('dt',                                   None)
        voxel_dims = float(kwargs.pop('voxel_dims',             None))
        buffer = float(kwargs.pop('buffer',                     None))
        path_to_bvals = kwargs.pop('path_to_bvals',             None)
        path_to_bvecs = kwargs.pop('path_to_bvecs',             None)
        random_state = kwargs.pop('random_state',               42  )

        if not isinstance(num_spins, int) or num_spins < 0:
            raise ValueError("Incorrect data type or value for spin. To run the simulation,"
                             +" make sure num_spins is an interger"
                             +" make sure num_spins > 0" 
                            )
        if not isinstance(fiber_fraction, tuple):
            raise ValueError("Incorrect Data Type for fiber_fraction. To run the simulation,"
                             +" make sure fiber_fraction is a tuple of the form: ([X, Y])"
                            )
        if not isinstance(fiber_radii, float):
            raise ValueError("Incorrect Data Type for fiber_radius. To run the simulation,"
                             +" make sure fiber_radius is a float"
                            )
        if not isinstance(thetas, tuple):
            raise ValueError("Incorrect Data Type for Thetas. To run the simulation,"
                            +" make sure Thetas is a tuple of the form: ([0, X])"
                            )
        if not isinstance(fiber_diffusions, tuple):
            raise ValueError("Incorrect Data Type for fiberDiffusions. To run the simulation,"
                             +" make sure fiberDiffusions is a tuple of the form: ([X, Y])"
                            )

        if not isinstance(cell_fraction, float):
            raise ValueError("Incorect Data Type for cellFraction. To run the simulation,"
                             +" make sure cellFraction is a float")
        
        if not isinstance(cell_radii, tuple):
             raise ValueError("Incorrect Data Type for cellRadii. To run the simulation,"
                            +" make sure cellRadii is a tuple of the form: ([X, Y])"
                            )
        if not isinstance(fiber_configuration, str):
             raise ValueError("Incorrect Data Type for fiberConfiguration. To run the simulation,"
                            +" make sure fiberConfiguration is a string"
                            )
        if not isinstance(Delta,float):
            raise ValueError("Incorrect Data Type for Delta. To run the simulation,"
                    +" make sure Delta is a float")

        if not isinstance(dt, float):
            raise ValueError("Inccorect Data Type for dt. To run the simulation,"
                    +" make sure cellFraction is a float")

        if not isinstance(voxel_dims, float):
            raise ValueError("Incorect Data Type for voxelDim. To run the simulation,"
                    +" make sure cellFraction is a float")

        if not isinstance(buffer, float):
            raise ValueError("Incorrect Data Type for buffer. To run the simulation,"
                    +" make sure cellFraction is a float")

        if not os.path.exists(path_to_bvals):
            raise ValueError("Path to bval files does not exist. To run the simulation,"
                             +" make sure you have entered a valid path to the bval file")
        if not os.path.exists(path_to_bvecs):
            raise ValueError("Path to bvec files does not exist. To run the simulation,"
                             +" make sure you have entered a valid path to the bvec file")
        if not os.path.exists(self.path_to_save):
            raise ValueError("Path to data directory does not exist. To run the simulation,"
                             +" make sure you have entered a valid path to the data directory")
        
        sys.stdout.write('\n    Inputs are valid!\n    Proceeding to simulation step.')

        data_dir = self.path_to_save + os.sep + "R=" + str(self.cfg_path).split('_Co',1)[0][-2] + "_C=" + str(self.cfg_path).split('_Co',1)[0][-1]
        if not os.path.exists(data_dir): os.mkdir(data_dir)
        path, file = os.path.split(self.cfg_path)  
        if not os.path.exists(data_dir + os.sep + file): shutil.move(self.cfg_path, data_dir + os.sep + file)
        np.random.seed(random_state)
        self.bvals = np.loadtxt(path_to_bvals) 
        self.bvecs = np.loadtxt(path_to_bvecs)
        self.voxelDims = voxel_dims      
        self.buffer = buffer 
        self.numSpins = num_spins
        self.fiberFraction = fiber_fraction
        self.fiberRadius = fiber_radii
        self.Thetas = thetas
        self.fiberDiffusions = fiber_diffusions
        self.fiberRotationReference = np.load(data_dir + os.sep + "rotReference.npy")
        self.rotMat = np.load(data_dir + os.sep + "rotMatrix.npy")      
        
        self.cellFraction = cell_fraction
        self.cellRadii = cell_radii
        self.fiberCofiguration = fiber_configuration
        self.voidDist = .60*self.voxelDims
        
        self.Delta = Delta
        self.dt = 0.001
        self.delta = 0.001
        self.fiberCenters = np.load(data_dir + os.sep + "fiberCenters.npy")
        self.cellCenters = np.load(data_dir + os.sep + "cellsCenters.npy")
        self.spinPositionsT1m = np.random.uniform(low = 0 , high = self.voxelDims, size = (int(self.numSpins),3))
        self.spinInFiber1_i = np.load(data_dir + os.sep + "indFiber1Spins.npy")
        self.spinInFiber2_i = np.load(data_dir + os.sep + "indFiber2Spins.npy")
        self.spinInCell_i = np.load(data_dir + os.sep + "indCellsSpins.npy")

        self.numFibers = len(self.fiberCenters)
        self.numCells = len(self.cellCenters)
        

    def _set_params_from_config(self, path_to_configuration_file):
        
        self.cfg_path = path_to_configuration_file
        ## Simulation Parameters
        config = configparser.ConfigParser()
        config.read(path_to_configuration_file)
        num_spins = literal_eval(config['Simulation Parameters']['numSpins'])
        fiber_fractions = literal_eval(config['Simulation Parameters']['fiberFraction'])
        fiber_radii = literal_eval(config['Simulation Parameters']['fiberRadius'])
        thetas = literal_eval(config['Simulation Parameters']['Thetas'])
        fiber_diffusions = literal_eval(config['Simulation Parameters']['fiberDiffusions'])
        cell_fraction = literal_eval(config['Simulation Parameters']['cellFraction'])
        cell_radii = literal_eval(config['Simulation Parameters']['cellRadii'])
        fiber_configuration = (config['Simulation Parameters']['fiberConfiguration'])
        
        ## Scanning Parameters
        Delta = literal_eval(config['Scanning Parameters']['Delta'])
        dt = literal_eval(config['Scanning Parameters']['dt'])
        voxel_dims = literal_eval(config['Scanning Parameters']['voxelDim'])
        buffer = literal_eval(config['Scanning Parameters']['buffer'])
        #bvals_path = config['Scanning Parameters']['path_to_bvals']
        #bvecs_path = config['Scanning Parameters']['path_to_bvecs']
        bvals_path = r"/bmr207/nmrgrp/nmr202/MCSIM/Repo/DBSI/bval-99.bval"
        bvecs_path = r"/bmr207/nmrgrp/nmr202/MCSIM/Repo/DBSI/bvec-99.bvec"
        ## Saving Parameters
        self.path_to_save = r"/bmr207/nmrgrp/nmr202/MCSIM/simOnlyTest"

        self.set_parameters(
            num_spins=num_spins,
            fiber_fraction= fiber_fractions,
            fiber_radius=fiber_radii,
            thetas=thetas,
            fiber_diffusions=fiber_diffusions,
            cell_fraction=cell_fraction,
            cell_radii=cell_radii,
            fiber_configuration=fiber_configuration,
            Delta=Delta,
            dt=dt,
            voxel_dims=voxel_dims,
            buffer=buffer,
            path_to_bvals= bvals_path, 
            path_to_bvecs= bvecs_path)
        return

    def from_config(self, path_to_configuration_file):
        self._set_params_from_config(path_to_configuration_file)    
        spin_positions_t2p,spin_positions_t1m = diffusion._simulate_diffusion(self.spinPositionsT1m, 
                                                                              self.spinInFiber1_i,
                                                                              self.spinInFiber2_i, 
                                                                              self.spinInCell_i,
                                                                              self.fiberCenters,
                                                                              self.cellCenters,
                                                                              self.Delta,
                                                                              self.dt,
                                                                              self.fiberCofiguration,
                                                                              self.fiberRotationReference)
        
        self.fiber1PositionsT1m = spin_positions_t1m[np.where(self.spinInFiber1_i > -1)]
        self.fiber1PositionsT2p = spin_positions_t2p[np.where(self.spinInFiber1_i > -1)]
        self.fiber2PositionsT1m = spin_positions_t1m[np.where(self.spinInFiber2_i > -1)]
        self.fiber2PositionsT2p = spin_positions_t2p[np.where(self.spinInFiber2_i > -1)]
        self.cellPositionsT1m   = spin_positions_t1m[np.where((self.spinInCell_i > -1) & (self.spinInFiber1_i == -1) & (self.spinInFiber2_i == -1))]
        self.cellPositionsT2p   = spin_positions_t2p[np.where((self.spinInCell_i > -1) & (self.spinInFiber1_i == -1) & (self.spinInFiber2_i == -1))] 
        self.extraPositionsT1m  = spin_positions_t1m[np.where((self.spinInCell_i == -1) & (self.spinInFiber1_i == -1) & (self.spinInFiber2_i == -1))] 
        self.extraPositionsT2p  = spin_positions_t2p[np.where((self.spinInCell_i == -1) & (self.spinInFiber1_i == -1) & (self.spinInFiber2_i == -1))] 
        
        sys.stdout.write('\n----------------------------------')
        sys.stdout.write('\n    Empirical Volume Fractions')
        sys.stdout.write('\n----------------------------------')
        sys.stdout.write('\n    Fiber 1 Volume: {}'.format(self.fiber1PositionsT1m.shape[0]/self.spinPositionsT1m.shape[0]))
        sys.stdout.write('\n    Fiber 2 Volume: {}'.format(self.fiber2PositionsT1m.shape[0]/self.spinPositionsT1m.shape[0]))
        sys.stdout.write('\nTotal Fiber Volume: {}'.format((self.fiber1PositionsT1m.shape[0] + self.fiber2PositionsT1m.shape[0])/self.spinPositionsT1m.shape[0]))
        sys.stdout.write('\n       Cell Volume: {}'.format(self.cellPositionsT1m.shape[0]/self.spinPositionsT1m.shape[0]))
        sys.stdout.write('\n      Water Volume: {}'.format(self.extraPositionsT1m.shape[0]/self.spinPositionsT1m.shape[0]))
        sys.stdout.write('\n\nProceeding to save results...')
        sys.stdout.write('\n')
        
        save_simulated_data._save_data(self, 
                                       self.path_to_save, 
                                       plot_xyz=False)
        return

    def _signal_from_trajectory_data(self,trajectory_dir):
        run_xyz = False
        finite  = False
        trajectory_t1ms = glob.glob(trajectory_dir + os.sep + '*T1m*.npy')
        for trajectory_file in trajectory_t1ms:
                traj_dir, fname = os.path.split(trajectory_file)
                compartment = (fname[0:5])
                
                traj1 = np.load(trajectory_file)
                traj2 = np.load(trajectory_file.replace('T1m', 'T2p'))
                if compartment == 'cellP':
                    compartment = 'cells'
                    self.cellPositionsT1m = traj1
                    self.cellPositionsT2p = traj2
                elif compartment == 'cells':
                    self.cellPositionsT1m = traj1
                    self.cellPositionsT2p = traj2

                if compartment == 'fiber':
                    self.fiberPositionsT1m = traj1
                    self.fiberPositionsT2p = traj2

                if compartment == 'water':
                    self.extraPositionsT1m = traj1
                    self.extraPositionsT2p = traj2

                signal, bvals = save_simulated_data._signal(self, traj1, traj2, run_xyz, finite)
                dwi = nb.Nifti1Image(signal.reshape(1,1,1,-1), affine = np.eye(4))
                nb.save(dwi, traj_dir + os.sep  + compartment + "Signal_30dir.nii")

        if self.cellFraction > 0:
            expSignal, bvals = save_simulated_data._signal(self, np.vstack([self.fiberPositionsT1m, self.cellPositionsT1m,self.extraPositionsT1m]), np.vstack([self.fiberPositionsT2p,self.cellPositionsT2p,self.extraPositionsT2p]), run_xyz, finite)
        else:
            expSignal, bvals = save_simulated_data._signal(self, np.vstack([self.fiberPositionsT1m,self.extraPositionsT1m]), np.vstack([self.fiberPositionsT2p,self.extraPositionsT2p]), run_xyz, finite)
        dwi = nb.Nifti1Image(expSignal.reshape(1,1,1,-1), affine = np.eye(4))
        nb.save(dwi, traj_dir + os.sep + "totalSignal_30dir.nii")
        return
        
def dmri_sim_wrapper(arg):
    path, file = os.path.split(arg)
    simObj = dmri_simulation()
    simObj.from_config(arg)

def main():       
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

    configs = glob.glob(r"/bmr207/nmrgrp/nmr202/MCSIM/simOnlyTest/*/*.ini")
    for cfg in configs:
        sys.stdout.write('\n-------------------------------------------------------------------')
        sys.stdout.write('\n                           Now Simulating:')
        sys.stdout.write('\n  {}'.format(str(cfg)))
        sys.stdout.write('\n-------------------------------------------------------------------')

        p = Process(target=dmri_sim_wrapper, args = (cfg,))
        p.start()
        p.join()


if __name__ == "__main__":
    mp.set_start_method('forkserver')
    main()
