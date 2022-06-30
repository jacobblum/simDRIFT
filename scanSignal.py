from re import I
import numpy as np 
import matplotlib.pyplot as plt
import scipy.io
import nnlsq_model
import nibabel as nb
import pandas as pd
import true_signal
from tqdm import tqdm 
import nnlsq_model 


def cleanSpins(trajectory_1, trajectory_2, spin_positions, spin_key):
    
    traj_1_in_voxel = []
    traj_2_in_voxel = []
    idxs = []

    trajectory_1 = trajectory_1[(spin_positions != 0) | (spin_positions != 0), :, :]
    trajectory_2 = trajectory_2[(spin_positions != 0) | (spin_positions != 0), :, :]

    for j in range(trajectory_1.shape[0]):
        t1j_min_pos, t1j_max_pos = np.amin(trajectory_1[j,:,:]), np.amax(trajectory_1[j,:,:])
        t2j_min_pos, t2j_max_pos = np.amin(trajectory_2[j,:,:]), np.amax(trajectory_2[j,:,:])

        if (t1j_min_pos >= 0 and t1j_max_pos <= 200) and (t2j_min_pos >= 0 and t2j_max_pos <= 200):
            traj_1_in_voxel.append(trajectory_1[j,:,:])
            traj_2_in_voxel.append(trajectory_2[j,:,:])
            idxs.append(j)

    outputarg1 = np.array(traj_1_in_voxel) 
    outputarg2 = np.array(traj_2_in_voxel)

    # 2802
    # 2224

    # 3913 cells 
    # 12122 water

    return outputarg1, outputarg2

def signal(trajectory_1, trajectory_2, gradients, bvals, spin_positions):
    gamma = 42.58
    delta = 6
    Delta = 18
    target_bmax = 1250 # s/mm^2 
    dt = 0.0050 #ms

    spin_loc_key = 4
    num_bvals = 20
    #b_vals = np.linspace(0,target_bmax, num_bvals)
    #Gx =  np.sqrt(10**-3 *b_vals/( gamma**2 * delta**2*(Delta-delta/3)))
    #unit_gradients = np.zeros((3*len(Gx),3))    
    #for i in range(3):
    #    unit_gradients[i*num_bvals: (i+1)*num_bvals,i] = Gx 

    unit_gradients = np.loadtxt(r"/Volumes/LaCie/MCSIM-Jacob/CrossingFibers/ABCD/sub-NDARDD890AYU_ses-01_dwi_sub-NDARDD890AYU_ses-01_dwi (1).bvec").T
    b_vals = (np.loadtxt(r'/Volumes/LaCie/MCSIM-Jacob/CrossingFibers/ABCD/sub-NDARDD890AYU_ses-01_dwi_sub-NDARDD890AYU_ses-01_dwi (3).bval'))
    all_signal = np.zeros(unit_gradients.shape[0])
    traj1_vox, traj2_vox = cleanSpins(trajectory_1, trajectory_2, spin_positions, spin_loc_key)
    for i in tqdm(range(len(unit_gradients[:,0]))):
        gradient_strength = np.sqrt((b_vals[i] * 10**-3)/(gamma**2*delta**2*(Delta-delta/3)))*unit_gradients[i,:]
        signal = 0
        for j in range(len(traj1_vox[:,0])):
            spin_i_traj_1 = traj1_vox[j,:,:]
            spin_i_traj_2 = traj2_vox[j,:,:]
            phase_shift = gamma * np.sum(gradient_strength.dot((spin_i_traj_2-spin_i_traj_1).T)) * dt     
            signal = signal + np.exp( -1 * (0+1j) * phase_shift)
       
        signal = signal/traj1_vox.shape[0]
        all_signal[i] = np.abs(signal)


    #fig, (ax1) = plt.subplots(figsize = (10,6))
    #ax1.plot(b_vals, all_signal[0:num_bvals], 'ro-', label = 'Cell Signal')
    #plt.legend()
    #plt.show()
    
    dwi_img = np.zeros((1,1,1,all_signal.shape[0]))
    dwi_img[0,0,0,:] = all_signal

    nifti_dwi = nb.Nifti1Image(dwi_img, affine = np.eye(4))
    nb.save(nifti_dwi, r"/Volumes/LaCie/MCSIM-Jacob/CrossingFibers/ABCD/no_ecef_Signal" + ".nii")
    #np.savetxt(r"D:\MCSIM-Jacob\CrossingFibers\DBSI-26\bval.bval", b_vals.T)
    return all_signal, b_vals, traj1_vox.shape[0]

def plot():
    #114
    #spin_positions_20k_114 = np.load(r'/Users/jacobblum/Desktop/MCSIM_Jacob/MCSIM-Jacob/CrossingFibers/Data/from114/220613/spin_positions_10%20k.npy')
    #spin_positions_25k_114 = np.load(r'/Users/jacobblum/Desktop/MCSIM_Jacob/MCSIM-Jacob/CrossingFibers/Data/from114/220613/spin_positions_10%25k.npy')
    #traj1_20k_114 = np.load(r'/Users/jacobblum/Desktop/MCSIM_Jacob/MCSIM-Jacob/CrossingFibers/Data/from114/220613/taj1_10%20k.npy')
    #traj1_25k_114 = np.load(r'/Users/jacobblum/Desktop/MCSIM_Jacob/MCSIM-Jacob/CrossingFibers/Data/from114/220613/taj1_10%25k.npy')
    #traj2_20k_114 = np.load(r'/Users/jacobblum/Desktop/MCSIM_Jacob/MCSIM-Jacob/CrossingFibers/Data/from114/220613/traj2_10%20k.npy')
    #traj2_25k_114 = np.load(r'/Users/jacobblum/Desktop/MCSIM_Jacob/MCSIM-Jacob/CrossingFibers/Data/from114/220613/traj2_10%25k.npy')

    #spin_positions_114 = np.hstack((spin_positions_20k_114, spin_positions_25k_114))
    #traj1_114 = np.vstack((traj1_20k_114, traj1_25k_114))
    #traj2_114 = np.vstack((traj2_20k_114, traj2_25k_114))



    #115
    #spin_positions_20k_115 = np.load(r'/Users/jacobblum/Desktop/MCSIM_Jacob/MCSIM-Jacob/CrossingFibers/Data/from115/220613/spin_positions_10%20k.npy')
    #spin_positions_25k_115 = np.load(r'/Users/jacobblum/Desktop/MCSIM_Jacob/MCSIM-Jacob/CrossingFibers/Data/from115/220613/spin_positions_10%25k.npy')
    #traj1_20k_115 = np.load(r'/Users/jacobblum/Desktop/MCSIM_Jacob/MCSIM-Jacob/CrossingFibers/Data/from115/220613/taj1_10%20k.npy')
    #traj1_25k_115 = np.load(r'/Users/jacobblum/Desktop/MCSIM_Jacob/MCSIM-Jacob/CrossingFibers/Data/from115/220613/taj1_10%25k.npy')
    #traj2_20k_115 = np.load(r'/Users/jacobblum/Desktop/MCSIM_Jacob/MCSIM-Jacob/CrossingFibers/Data/from115/220613/traj2_10%20k.npy')
    #traj2_25k_115 = np.load(r'/Users/jacobblum/Desktop/MCSIM_Jacob/MCSIM-Jacob/CrossingFibers/Data/from115/220613/traj2_10%25k.npy')

    #spin_positions_115 = np.hstack((spin_positions_20k_115, spin_positions_25k_115))
    #traj1_115 = np.vstack((traj1_20k_115, traj1_25k_115))
    #traj2_115 = np.vstack((traj2_20k_115, traj2_25k_115))

    #116

    #117
    #traj1 = np.vstack((traj1_114, traj1_115))
    #traj2 = np.vstack((traj2_114, traj2_115))
    #spin_positions = np.hstack((spin_positions_114, spin_positions_115))
    
    positions = np.load(r'/Volumes/LaCie/MCSIM-Jacob/CrossingFibers/Data/22_0624/spin_positions_10%25k.npy')
    traj1 = np.load(r"//Volumes/LaCie/MCSIM-Jacob/CrossingFibers/Data/22_0624/taj1_10%25k.npy")
    traj2 = np.load(r'/Volumes/LaCie/MCSIM-Jacob/CrossingFibers/Data/22_0624/traj2_10%25k.npy')
    all_signal, b_vals, num_spins = signal(traj1, traj2, 1, 1, positions)
    
    exit()
    
    sx, sy, sz = all_signal[0:20], all_signal[20:40], all_signal[40:60]
    
    Dx = nnlsq_model.lstq(b_vals, sx)
    Dy = nnlsq_model.lstq(b_vals, sy)
    Dz = nnlsq_model.lstq(b_vals, sz)

    fig, (ax1) = plt.subplots(1,1, figsize = (12,10))
    ax1.plot(b_vals, np.log(sx), 'rx', label = r'$ADC^{x}:$ %s' %str(round(Dx[1],3)))
    ax1.plot(b_vals, np.log(sy), 'bx', label = r'$ADC^{y}:$ %s' %str(round(Dy[1],3)))
    ax1.plot(b_vals, np.log(sz), 'gx', label = r'$ADC^{z}$ %s' %str(round(Dz[1],3)))
    ax1.set_xlabel(r'$b_{k} \quad s/mm^{2}$')
    ax1.set_ylabel(r'$s_{k}$')
    ax1.set_title(r' dMRI Signal Attenuation - AD = 1.0, nspins = %s' %num_spins)
    #ax1.set_ylim(bottom = 0, top =1.1)
    #ax2.vlines(d_x, ymin = np.zeros(d_x.shape[0]), ymax = cc_x, color = 'r', label = r'$s_{k}^{x}$')
    #ax2.vlines(d_y, ymin = np.zeros(d_y.shape[0]), ymax = cc_y, color = 'b', label = r'$s_{k}^{y}$')
    #ax2.vlines(d_z, ymin = np.zeros(d_z.shape[0]), ymax = cc_z, color = 'g', label = r'$s_{k}^{z}$')
    #ax2.set_xlabel(r' D $ \mu m^{2} / ms $')
    #ax2.set_ylabel(r'Diffusivity Fraction')
    #ax2.set_title(r'Diffusion Spectra')
    #ax2.set_ylim(bottom = 0,top = 1.1)
    #ax2.set_xlim(left = -.05, right = 3.6)
    ax1.legend()
    #ax2.legend(loc = 'center')
    plt.show()
    return 

plot()