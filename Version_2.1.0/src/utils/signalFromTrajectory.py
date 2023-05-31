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
        fiberPositionsT1m =         0.0
        fiberPositionsT2p =         0.0
        cellPositionsT1m =          0.0
        cellPositionsT2p =          0.0
        extraPositionsT1m =         0.0
        extraPositionsT2p =         0.0
        spinInFiber_i =             0.0
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
        self.bvals = np.loadtxt(path_to_bvals) 
        self.bvecs = np.loadtxt(path_to_bvecs)
        self.voxelDims = voxel_dims      
        self.buffer = buffer 
        self.numSpins = num_spins
        self.fiberFraction = fiber_fraction
        self.fiberRadius = fiber_radii
        self.Thetas = thetas
        self.fiberDiffusions = fiber_diffusions
        self.Delta = Delta
        self.dt = dt
        self.delta = dt
        self.cellFraction = cell_fraction
        return 


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
        bvals_path = r"C:\MCSIM\Repo\simulation_data\DBSI\DBSI-30\bval-30.bval"
        bvecs_path = r"C:\MCSIM\Repo\simulation_data\DBSI\DBSI-30\bvec-30.bvec"
        
        ## Saving Parameters
        #self.path_to_save = config['Saving Parameters']['path_to_save_file_dir']
        self.path_to_save = r"C:\Users\kainen.utt\Documents\MATLAB\Mosaic_Plots\input\9x9"

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

def sig_from_traj(arg):
    path, file = os.path.split(arg)
    genSig = dmri_simulation()
    genSig._set_params_from_config(arg)
    genSig._signal_from_trajectory_data(path)

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

    configs = glob.glob(r"C:\Users\kainen.utt\Documents\MATLAB\Mosaic_Plots\input\9x9\*\*_*.ini")
    for cfg in configs:
        print('\nNow generating signal from trajectory data for: ' + str(cfg).split(r'9x9',1)[1][1:8])
        sig_from_traj(cfg,)


if __name__ == "__main__":
    main()
