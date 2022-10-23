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


def _add_noise(self, signal, snr, noise_type):
    sigma = 1.0/snr 
    real_channel_noise = np.random.normal(0, sigma, signal.shape[0])
    complex_channel_noise = np.random.normal(0, sigma, signal.shape[0])
    if noise_type == 'Rician':
        return np.sqrt((signal + real_channel_noise)**2 + complex_channel_noise**2)
    if noise_type == 'Gaussian':
        return signal + real_channel_noise


def _signal(self, trajectoryT1m, trajectoryT2p, xyz, finite):
        """
        Aquire the signal by integrating the ensemble distribution from t1m to t2p; int

        Parameters
        ----------
        trajectoryT1m: N_{spins} x 3 ndarray
            The initial spin position at time t1m
        
        trajectoryT2p: N_{spins} x 3 ndarray
            The spin position at time t2p

        Returns
        -------
        allSignal: (N_{bvals}, ) ndarray
            The signal induced by the k-th diffusion gradient and diffusion weighting factor
        
        b_vals: (N_{bvals},) ndarray
            The b-values used in the diffusion experiment

        Notes
        -----
        None
        
        References
        ----------
        [1] ... Rafael-Patino et. al. (2020)  Robust Monte-Carlo Simulations in Diffusion-MRI: 
                Effect of the Substrate Complexity and Parameter Choice on the Reproducibility of Results, 
                Front. Neuroinform., 10 March 2020 
        
        """
        
        gamma = 42.58
        Delta = self.Delta #ms
        dt = self.delta # ms 
        delta = dt #ms
        b_vals = np.linspace(0, 2200, 20)
        if finite: trajectoryT1m, trajectoryT2p = self.spins_in_voxel(trajectoryT1m, trajectoryT2p)
        if xyz:
            Gt = np.sqrt(10**-3 * b_vals/(gamma**2 * delta**2*(Delta-delta/3)))
            unitGradients = np.zeros((3*len(Gt), 3))
            for i in (range(unitGradients.shape[1])):
                unitGradients[i*len(b_vals): (i+1)*len(b_vals),i] = Gt
        else:
            unitGradients = self.bvecs.T
            b_vals = self.bvals
        allSignal = np.zeros(unitGradients.shape[0])
        for i in (range(unitGradients.shape[0])):
            sys.stdout.write('\r' + 'Gradient: ' +  str(i+1) + '/' + str(unitGradients.shape[0]))
            sys.stdout.flush()
            signal = 0
            if xyz:
                scaledGradient = unitGradients[i,:]
            else:
                scaledGradient = np.sqrt( (b_vals[i] * 10**-3)/ (gamma**2*delta**2*(Delta - delta/3))) * unitGradients[i,:]
            for j in range(trajectoryT1m.shape[0]):
                phase_shift = gamma * np.sum(scaledGradient.dot(trajectoryT1m[j,:]-trajectoryT2p[j,:])) * dt
                signal = signal + np.exp(-1 *(0+1j) * phase_shift)
            signal = signal/trajectoryT1m.shape[0]
            allSignal[i] = np.abs(signal)
        dwi = nb.Nifti1Image(allSignal.reshape(1,1,1,-1), affine = np.eye(4))
        return allSignal, b_vals

def _save_data(self, path, plot_xyz):
    data_dir = path + os.sep + "FF={}_CF={}_CellRad={}_Theta={}_Diffusions={}_fibConfig={}_Sim".format(self.fiberFraction, self.cellFraction, self.cellRadii, self.Thetas, self.fiberDiffusions, self.fiberCofiguration)
    if not os.path.exists(data_dir): os.mkdir(data_dir)
    path, file = os.path.split(self.cfg_path)  
    if not os.path.exists(data_dir + os.sep + file): shutil.move(self.cfg_path, data_dir + os.sep + file)
    overallData = []
   
    fiber_trajectories = [self.fiberPositionsT1m, self.fiberPositionsT2p]
    overallData.append(fiber_trajectories)
    np.save(data_dir + os.sep + "fiberPositionsT1m_angle={}_diffusivities={}_dt={}_ff={}.npy".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)), self.fiberPositionsT1m)
    np.save(data_dir + os.sep + "fiberPositionsT2p_angle={}_diffusivities={}_dt={}_ff={}.npy".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)), self.fiberPositionsT2p)
    sys.stdout.write('\nAquiring Pure Fiber Signal: \n')
    pureFiberSignal, b = _signal(self, self.fiberPositionsT1m, self.fiberPositionsT2p, xyz = False, finite = False)
    dwiFiber = nb.Nifti1Image(pureFiberSignal.reshape(1,1,1,-1), affine = np.eye(4))
    nb.save(dwiFiber, data_dir + os.sep + "pureFiberSignal_angle={}_diffusivities={}_dt={}_ff={}.nii".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)))


    cell_trajectories = [self.cellPositionsT1m, self.cellPositionsT2p]
    overallData.append(cell_trajectories)
    np.save(data_dir + os.sep + "cellPositionsT1m_angle={}_diffusivities={}_dt={}_ff={}.npy".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)), self.cellPositionsT1m)
    np.save(data_dir + os.sep + "cellPositionsT2p_angle={}_diffusivities={}_dt={}_ff={}.npy".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)), self.cellPositionsT2p)
    sys.stdout.write('\nAquiring Pure Cell Signal: \n')
    pureCellSignal, _ = _signal(self,self.cellPositionsT1m, self.cellPositionsT2p, xyz = False, finite=False)
    dwiCell = nb.Nifti1Image(pureCellSignal.reshape(1,1,1,-1), affine = np.eye(4))
    nb.save(dwiCell, data_dir + os.sep + "pureCellSignal_angle={}_diffusivities={}_dt={}_ff={}.nii".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)))

    water_trajectories = [self.extraPositionsT1m, self.extraPositionsT2p]
    overallData.append(water_trajectories)
    np.save(data_dir + os.sep + "waterPositionsT1m_angle={}_diffusivities={}_dt={}_ff={}.npy".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)), self.extraPositionsT1m)
    np.save(data_dir + os.sep + "waterPositionsT2p_angle={}_diffusivities={}_dt={}_ff={}.npy".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)), self.extraPositionsT2p)
    sys.stdout.write('\nAquiring Pure Extra Cell/Fiber Signal: \n')
    pureWaterSignal, bvals = _signal(self,self.extraPositionsT1m, self.extraPositionsT2p, xyz = False,finite=False)
    dwiWater = nb.Nifti1Image(pureWaterSignal.reshape(1,1,1,-1), affine = np.eye(4))
    nb.save(dwiWater, data_dir + os.sep  + "pureWaterSignal_angle={}_diffusivities={}_dt={}_ff={}.nii".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)))

    sys.stdout.write('\nAquiring Total Signal: \n')
    expSignal, bvals = _signal(self, np.vstack([self.fiberPositionsT1m, self.cellPositionsT1m,self.extraPositionsT1m]), np.vstack([self.fiberPositionsT2p,self.cellPositionsT2p,self.extraPositionsT2p]), xyz = False, finite = False)
    dwi = nb.Nifti1Image(expSignal.reshape(1,1,1,-1), affine = np.eye(4))
    nb.save(dwi,data_dir + os.sep + "totalSignal_angle={}_diffusivities={}_dt={}_ff={}.nii".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)))

    return