import numpy as np 
import numba 
from numba import jit, cuda, int32, float32
from numba.cuda import random 
from numba.cuda.random import xoroshiro128p_normal_float32,  create_xoroshiro128p_states
import math
import jp
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
        numSpins = 0.0 
        numFibers = 0.0
        fiberFraction = 0.0
        fiberRadius = 0.0 #um
        Delta = 0.0 #ms
        dt = 0.0 #ms
        delta = 0.0 #ms
        numCells = 0.0
        cellFraction = 0.0
        cellRadii = 0.0 #um
        spinPositionsT1m = 0
        fiberPositionsT1m = 0
        fiberPositionsT2p = 0
        cellPositionsT1m = 0
        cellPositionsT2p = 0
        extraPositionsT1m = 0
        extraPositionsT2p = 0
        spinInFiber_i = 0
        spinInCell_i = 0
        fiberRotationReference = 0
        Thetas = 0
        fiberDiffusions  = 0
        rotMat = 0
        simulateFibers = 0
        simulateCells = 0
        simulateExtra = 0
        fiberConfiguration = ''
        voidDist = 0
        buffer = 100
        bvals = 0
        bvecs = 0
        cfg_path = ''
        path_to_save = ''

        return
    
    def set_parameters(self, numSpins, fiberFraction, fiberRadius, Thetas, fiberDiffusions, cellFraction, cellRadii, fiberConfiguration, Delta, dt, voxelDim, buffer, path_to_bvals, path_to_bvecs):
        self.bvals = np.loadtxt(path_to_bvals) 
        self.bvecs = np.loadtxt(path_to_bvecs)
        self.voxelDims = voxelDim
        self.buffer = buffer 
        self.numSpins = numSpins
        self.fiberFraction = fiberFraction
        self.fiberRadius = fiberRadius
        self.Thetas = Thetas
        self.fiberDiffusions = fiberDiffusions
        self.fiberRotationReference, self.rotMat = set_voxel_configuration._generate_rot_mat(Thetas)        
        self.numFibers = set_voxel_configuration._set_num_fibers(self.fiberFraction, self.fiberRadius,self.voxelDims, self.buffer)
        self.cellFraction = cellFraction
        self.cellRadii = cellRadii
        self.fiberCofiguration = fiberConfiguration
        self.voidDist = .60*self.voxelDims
        self.numCells = set_voxel_configuration._set_num_cells(self.cellFraction, self.cellRadii, self.voxelDims, self.buffer)
        self.Delta = Delta
        self.dt = dt
        self.delta = dt
        self.fiberCenters = set_voxel_configuration._place_fiber_grid(self.fiberFraction, self.numFibers, self.fiberRadius, self.fiberDiffusions, self.voxelDims, self.buffer, self.voidDist, self.rotMat, self.fiberCofiguration)
        self.cellCenters = set_voxel_configuration._place_cells(self.numCells, self.cellRadii, self.fiberCofiguration, self.voxelDims, self.buffer, self.voidDist)
        self.spinPotionsT1m = np.random.uniform(low = 0 , high = 60, size = (int(self.numSpins),3))
        self.spinInFiber_i, self.spinInCell_i = spin_init_positions._find_spin_locations(self.spinPotionsT1m, self.fiberCenters, self.cellCenters, self.fiberRotationReference )
        
    def _set_params_from_config(self, path_to_configuration_file):
        self.cfg_path = path_to_configuration_file
        ## Simulation Parameters
        config = configparser.ConfigParser()
        config.read(path_to_configuration_file)
        numSpins = literal_eval(config['Simulation Parameters']['numSpins'])
        fiberFraction = literal_eval(config['Simulation Parameters']['fiberFraction'])
        fiberRadius = literal_eval(config['Simulation Parameters']['fiberRadius'])
        Thetas = literal_eval(config['Simulation Parameters']['Thetas'])
        fiberDiffusions = literal_eval(config['Simulation Parameters']['fiberDiffusions'])
        cellFraction = literal_eval(config['Simulation Parameters']['cellFraction'])
        cellRadii = literal_eval(config['Simulation Parameters']['cellRadii'])
        fiberConfiguration = (config['Simulation Parameters']['fiberConfiguration'])
        self.simulateFibers = literal_eval(config['Simulation Parameters']['simulateFibers'])
        self.simulateCells = literal_eval(config['Simulation Parameters']['simulateCells'])
        self.simulateExtra = literal_eval(config['Simulation Parameters']['simulateExtraEnvironment'])

        ## Scanning Parameters
        Delta = literal_eval(config['Scanning Parameters']['Delta'])
        dt = literal_eval(config['Scanning Parameters']['dt'])
        voxelDims = literal_eval(config['Scanning Parameters']['voxelDim'])
        buffer = literal_eval(config['Scanning Parameters']['buffer'])
        bvals_path = config['Scanning Parameters']['path_to_bvals']
        bvecs_path = config['Scanning Parameters']['path_to_bvecs']

        ## Saving Parameters
        self.path_to_save = config['Saving Parameters']['path_to_save_file_dir']

        self.set_parameters(
            numSpins=numSpins,
            fiberFraction= fiberFraction,
            fiberRadius=fiberRadius,
            Thetas=Thetas,
            fiberDiffusions=fiberDiffusions,
            cellFraction=cellFraction,
            cellRadii=cellRadii,
            fiberConfiguration=fiberConfiguration,
            Delta=Delta,
            dt=dt,
            voxelDim=voxelDims,
            buffer=buffer,
            path_to_bvals= bvals_path, 
            path_to_bvecs= bvecs_path)
        return

    def from_config(self, path_to_configuration_file):
        self._set_params_from_config(path_to_configuration_file)
        spin_positions_t2p,spin_positions_t1m = diffusion._simulate_diffusion(self.spinPotionsT1m, 
                                      self.spinInFiber_i, 
                                      self.spinInCell_i,
                                      self.fiberCenters,
                                      self.cellCenters,
                                      self.Delta,
                                      self.dt,
                                      self.fiberCofiguration,
                                      self.fiberRotationReference)
        
        self.fiberPositionsT1m  = spin_positions_t1m[np.where(self.spinInFiber_i > -1)]
        self. fiberPositionsT2p = spin_positions_t2p[np.where(self.spinInFiber_i > -1)]
        self.cellPositionsT1m   = spin_positions_t1m[np.where((self.spinInCell_i > -1) & (self.spinInFiber_i == -1))]
        self.cellPositionsT2p   = spin_positions_t2p[np.where((self.spinInCell_i > -1) & (self.spinInFiber_i == -1))] 
        self.extraPositionsT1m  = spin_positions_t1m[np.where((self.spinInCell_i == -1) & (self.spinInFiber_i == -1))]
        self.extraPositionsT2p  = spin_positions_t2p[np.where((self.spinInCell_i == -1) & (self.spinInFiber_i == -1))]
        
        sys.stdout.write('\nMC Integration Empirical Volume Fractions:')
        sys.stdout.write('\nFiber Volume: {}'.format(self.fiberPositionsT1m.shape[0]/self.spinPotionsT1m.shape[0]))
        sys.stdout.write('\nCell Volume: {}'.format(self.cellPositionsT1m.shape[0]/self.spinPotionsT1m.shape[0]))
        sys.stdout.write('\nExtra Cellular/Fiber Volume: {}'.format(self.extraPositionsT1m.shape[0]/self.spinPotionsT1m.shape[0]))
        sys.stdout.write('\n\nSaving Results...')
        sys.stdout.write('\n')
        save_simulated_data._save_data(self, self.path_to_save, plot_xyz=False)
        return
    
    def _signal_from_trajectory_data(self,trajectory_dir):
        trajectory_t1ms = glob.glob(trajectory_dir + os.sep + '*water*T1m*.npy')
        for trajectory_file in trajectory_t1ms:
                for f in ['fin', 'inf']:
                    traj_dir, fname = os.path.split(trajectory_file)
                    compartment = (fname[0:5])
                    traj1 = np.load(trajectory_file)
                    traj2 = np.load(trajectory_file.replace('T1m', 'T2p'))
                    fig, ax = plt.subplots(figsize = (10,3))
                    ax.hist(traj2-traj1, bins = 1000)
                    plt.show()
                    signal, bvals = self.signal(traj1, traj2, xyz = False, finite = (f == 'fin'))
                    plt.show()
                    dwi = nb.Nifti1Image(signal.reshape(1,1,1,-1), affine = np.eye(4))
                    nb.save(dwi, traj_dir + os.sep  + compartment + '_' + f + "_Signal_angle={}_diffusivities={}_dt={}_ff={}.nii".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)))
        return
def dmri_sim_wraper(arg):
    path, file = os.path.split(arg)
    x = dmri_simulation()
    x.from_config(arg)
    #x._set_params_from_config(arg)
    #x._signal_from_trajectory_data(path)
    
def main():       
    #numba.cuda.detect()
    configs = glob.glob(r"C:\MCSIM\dMRI-MCSIM-main\Yes_Cells\R4_config_Theta=(0, 90)_fibFrac=(0.1, 0.1)_cellFrac=0.35_cellRad=(2.5, 2.5)_Diff=(1.0, 2.0)_Pene.ini")
    for cfg in configs:
        p = Process(target=dmri_sim_wraper, args = (cfg,))
        p.start()
        p.join()
    
    """
    sim = dmri_simulation()
    sim.set_parameters(
        numSpins= 100*10**3,
        fiberFraction= (0.50, .50),  # Fraction in each Half/Quadrant Depending on 'P'/'NP'
        fiberRadius= 1.0,            # um
        Thetas = (0,90),              # degrees
        fiberDiffusions= (1.0, 1.0), # um^2/mm
        cellFraction= .10,            # Fraction in each Half/Quadrant Depending on 'P'/'NP'
        cellRadii= (10,5),           # um
        fiberConfiguration = 'Void',           # 'P' = Penetrating Cells; 'NP = Non-Penetrating Cells, 'IW' 
        Delta = 10,                  # ms  
        dt = 0.001,                  # ms 
        voxelDim= 50,                # um
        buffer = 0,                 # um
        path_to_bvals= r'C:\MCSIM\Repo\simulation_data\DBSI\DBSI-99\bval\bval-99.bval',
        path_to_bvecs= r'C:\MCSIM\Repo\simulation_data\DBSI\DBSI-99\bvec\bvec-99.bvec',
        path_to_save_file_dir= r'C:\MCSIM\dMRI-MCSIM-main\klu_test'
        )   
    """


if __name__ == "__main__":
    main()
    
 



