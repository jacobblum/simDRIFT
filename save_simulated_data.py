import numpy as np 
import numba 
from numba import jit, cuda, int32, float32
from numba.cuda import random 
from numba.cuda.random import xoroshiro128p_normal_float32,  create_xoroshiro128p_states
import math
import jp as jp
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
        b_vals = np.linspace(0, 2000, 20)
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
    data_dir = path + os.sep + "R=" + str(self.cfg_path).split('_Config',1)[0][-2] + "_C=" + str(self.cfg_path).split('_Config',1)[0][-1]
    
    if not os.path.exists(data_dir): os.mkdir(data_dir)
    path, file = os.path.split(self.cfg_path)  
    if not os.path.exists(data_dir + os.sep + file): shutil.move(self.cfg_path, data_dir + os.sep + file)
    
    overallData = []
    sys.stdout.write('\nAquiring Pure Fiber Signal: \n')
   
    if self.fiberFraction[0] > 0:
        fiber1_trajectories = [self.fiber1PositionsT1m, self.fiber1PositionsT2p]
        overallData.append(fiber1_trajectories)
        np.save(data_dir + os.sep + "fiber1PositionsT1m.npy", self.fiber1PositionsT1m)
        np.save(data_dir + os.sep + "fiber1PositionsT2p.npy", self.fiber1PositionsT2p)
        pureFiber1Signal, b = _signal(self, self.fiber1PositionsT1m, self.fiber1PositionsT2p, xyz = False, finite = False)
        dwiFiber1 = nb.Nifti1Image(pureFiber1Signal.reshape(1,1,1,-1), affine = np.eye(4))
        nb.save(dwiFiber1, data_dir + os.sep + "pureFiber1Signal.nii")

    if self.fiberFraction[1] > 0:
        fiber2_trajectories = [self.fiber2PositionsT1m, self.fiber2PositionsT2p]
        overallData.append(fiber2_trajectories)
        np.save(data_dir + os.sep + "fiber2PositionsT1m.npy", self.fiber2PositionsT1m)
        np.save(data_dir + os.sep + "fiber2PositionsT2p.npy", self.fiber2PositionsT2p)
        pureFiber2Signal, b = _signal(self, self.fiber2PositionsT1m, self.fiber2PositionsT2p, xyz = False, finite = False)
        dwiFiber2 = nb.Nifti1Image(pureFiber2Signal.reshape(1,1,1,-1), affine = np.eye(4))
        nb.save(dwiFiber2, data_dir + os.sep + "pureFiber2Signal.nii")

    if (self.fiberFraction[0] > 0) and (self.fiberFraction[1] > 0):
        np.save(data_dir + os.sep + "fiberPositionsT1m.npy",np.vstack([self.fiber1PositionsT1m, self.fiber2PositionsT1m]))
        np.save(data_dir + os.sep + "fiberPositionsT2p.npy",np.vstack([self.fiber1PositionsT2p, self.fiber2PositionsT2p]))
        comboFiberSignal, b = _signal(self, np.vstack([self.fiber1PositionsT1m, self.fiber2PositionsT1m]), np.vstack([self.fiber1PositionsT2p, self.fiber2PositionsT2p]), xyz = False, finite = False)
        dwiFibers = nb.Nifti1Image(comboFiberSignal.reshape(1,1,1,-1), affine = np.eye(4))
        nb.save(dwiFibers, data_dir + os.sep + "comboFiberSignal.nii")
    
    if self.cellFraction > 0:
        cell_trajectories = [self.cellPositionsT1m, self.cellPositionsT2p]
        overallData.append(cell_trajectories)
        np.save(data_dir + os.sep + "cellsPositionsT1m.npy", self.cellPositionsT1m)
        np.save(data_dir + os.sep + "cellsPositionsT2p.npy", self.cellPositionsT2p)
        sys.stdout.write('\nAquiring Pure Cell Signal: \n')
        pureCellSignal, _ = _signal(self,self.cellPositionsT1m, self.cellPositionsT2p, xyz = False , finite=False)
        dwiCell = nb.Nifti1Image(pureCellSignal.reshape(1,1,1,-1), affine = np.eye(4))
        nb.save(dwiCell, data_dir + os.sep + "pureCellsSignal.nii")

    water_trajectories = [self.extraPositionsT1m, self.extraPositionsT2p]
    overallData.append(water_trajectories)
    np.save(data_dir + os.sep + "waterPositionsT1m.npy", self.extraPositionsT1m)
    np.save(data_dir + os.sep + "waterPositionsT2p.npy", self.extraPositionsT2p)
    sys.stdout.write('\nAquiring Pure Extra Cell/Fiber Signal: \n')
    pureWaterSignal, bvals = _signal(self,self.extraPositionsT1m, self.extraPositionsT2p, xyz = False, finite=False)
    dwiWater = nb.Nifti1Image(pureWaterSignal.reshape(1,1,1,-1), affine = np.eye(4))
    nb.save(dwiWater, data_dir + os.sep  + "pureWaterSignal.nii")

    sys.stdout.write('\nAquiring Total Signal: \n')
    if (self.cellFraction > 0):
        if (self.fiberFraction[0] > 0) and (self.fiberFraction[1] > 0):
            expSignal, bvals = _signal(self, np.vstack([self.fiber1PositionsT1m, self.fiber2PositionsT1m, self.cellPositionsT1m,self.extraPositionsT1m]), np.vstack([self.fiber1PositionsT2p, self.fiber2PositionsT2p, self.cellPositionsT2p, self.extraPositionsT2p]), xyz = False, finite = False)
        elif (self.fiberFraction[0] > 0) and (self.fiberFraction[1] <= 0):
            expSignal, bvals = _signal(self, np.vstack([self.fiber1PositionsT1m, self.cellPositionsT1m,self.extraPositionsT1m]), np.vstack([self.fiber1PositionsT2p, self.cellPositionsT2p, self.extraPositionsT2p]), xyz = False, finite = False)
        elif (self.fiberFraction[0] <= 0) and (self.fiberFraction[1] <= 0):
            expSignal, bvals = _signal(self, np.vstack([self.cellPositionsT1m, self.extraPositionsT1m]), np.vstack([self.cellPositionsT2p, self.extraPositionsT2p]), xyz = False, finite = False)
    elif (self.cellFraction <= 0):
        if (self.fiberFraction[0] > 0) and (self.fiberFraction[1] > 0):
            expSignal, bvals = _signal(self, np.vstack([self.fiber1PositionsT1m, self.fiber2PositionsT1m,self.extraPositionsT1m]), np.vstack([self.fiber1PositionsT2p, self.fiber2PositionsT2p, self.extraPositionsT2p]), xyz = False, finite = False)
        elif (self.fiberFraction[0] > 0) and (self.fiberFraction[1] <= 0):
            expSignal, bvals = _signal(self, np.vstack([self.fiber1PositionsT1m, self.extraPositionsT1m]), np.vstack([self.fiber1PositionsT2p, self.extraPositionsT2p]), xyz = False, finite = False)
        elif (self.fiberFraction[0] <= 0) and (self.fiberFraction[1] <= 0):
            expSignal, bvals = _signal(self, self.extraPositionsT1m, self.extraPositionsT2p, xyz = False, finite = False)
    
    dwi = nb.Nifti1Image(expSignal.reshape(1,1,1,-1), affine = np.eye(4))
    nb.save(dwi,data_dir + os.sep + "totalSignal.nii")
    return
