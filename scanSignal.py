from re import I
import numpy as np 
import matplotlib.pyplot as plt
import scipy.io
import nnlsq_model
import nibabel as nb
import pandas as pd
import true_signal


def cleanSpins(trajectory_1, trajectory_2, spin_positions, spin_key):
    
    traj_1_in_voxel = []
    traj_2_in_voxel = []
    idxs = []

    trajectory_1 = trajectory_1[spin_positions <= spin_key, :, :]
    trajectory_2 = trajectory_2[spin_positions <= spin_key, :, :]


    for j in range(trajectory_1.shape[0]):
        t1j_min_pos, t1j_max_pos = np.amin(trajectory_1[j,:,:]), np.amax(trajectory_1[j,:,:])
        t2j_min_pos, t2j_max_pos = np.amin(trajectory_2[j,:,:]), np.amax(trajectory_2[j,:,:])


        if (t1j_min_pos >= 0 and t1j_max_pos <= 200) and (t2j_min_pos >= 0 and t2j_max_pos <= 200):
            traj_1_in_voxel.append(trajectory_1[j,:,:])
            traj_2_in_voxel.append(trajectory_2[j,:,:])
            idxs.append(j)

    
    outputarg1 = np.array(traj_1_in_voxel) 
    outputarg2 = np.array(traj_2_in_voxel)


    print(outputarg2.shape)
    outputarg3 = np.array(idxs)

    return outputarg1, outputarg2, outputarg3


def signal(trajectory_1, trajectory_2, gradients, bvals, spin_positions):
    gamma = 42.58
    delta = 5
    Delta = 20
    target_bmax = 3000 # s/mm^2 
    dt = 0.0050 #ms

    spin_loc_key = 3

    num_bvals = 20

    b_vals = np.geomspace(100,target_bmax, num_bvals)
    Gx =  np.sqrt(10**-3 * b_vals * 1/( gamma**2 * delta**2 * (Delta-delta/3)))
    unit_gradients = np.zeros((num_bvals*len(Gx),3))    
    for i in range(3):
        unit_gradients[i*num_bvals: (i+1)*num_bvals,i] = Gx 

    unit_gradients = np.loadtxt(r"Z:\Jacob_Blum\MCSIM-Jacob\signals\CrossingFibers\ABCD\sub-NDARDD890AYU_ses-01_dwi_sub-NDARDD890AYU_ses-01_dwi (1).bvec").T
    b_vals = (np.loadtxt(r"Z:\Jacob_Blum\MCSIM-Jacob\signals\CrossingFibers\ABCD\sub-NDARDD890AYU_ses-01_dwi_sub-NDARDD890AYU_ses-01_dwi (3).bval"))
    
    
    all_signal = np.zeros(unit_gradients.shape[0])
    traj1_vox, traj2_vox, idxs = cleanSpins(trajectory_1, trajectory_2, spin_positions, spin_loc_key)

    for i in range(len(unit_gradients[:,0])):
        gradient_strength =  np.sqrt(10**-3 * b_vals[i]/(gamma**2 * delta**2 * (Delta-delta/3))) * unit_gradients[i,:]
        #gradient_strength = unit_gradients[i,:]
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
    nb.save(nifti_dwi, r"Z:\Jacob_Blum\MCSIM-Jacob\signals\CrossingFibers\ABCD\all_signal" + ".nii")
    #np.savetxt(r"Z:\Jacob_Blum\MCSIM-Jacob\signals\DBSI-26\bvec.bvec", unit_gradients.T)
    return all_signal, b_vals, traj1_vox.shape[0]

def plot():

    traj_11 = np.load(r"Z:\Jacob_Blum\MCSIM-Jacob\sim_data\from_114\220613\taj1_10%20k.npy")
    traj_21 = np.load(r"Z:\Jacob_Blum\MCSIM-Jacob\sim_data\from_114\220613\traj2_10%20k.npy")
    spin_positions_1 = np.load(r"Z:\Jacob_Blum\MCSIM-Jacob\sim_data\from_114\220613\spin_positions_10%20k.npy")

    traj_12 = np.load(r"Z:\Jacob_Blum\MCSIM-Jacob\sim_data\from_115\220613\taj1_10%20k.npy")
    traj_22 = np.load(r"Z:\Jacob_Blum\MCSIM-Jacob\sim_data\from_115\220613\traj2_10%20k.npy")
    spin_positions_2 = np.load(r"Z:\Jacob_Blum\MCSIM-Jacob\sim_data\from_115\220613\spin_positions_10%20k.npy") 

    traj_1 = np.vstack((traj_11, traj_12))
    traj_2 = np.vstack((traj_21, traj_22))
    spin_positions = np.append(spin_positions_1, spin_positions_2)

    all_signal, b_vals, num_spins = signal(traj_1, traj_2, 1, 1, spin_positions)

    exit()
    
    sx, sy, sz = all_signal[0:20], all_signal[20:40], all_signal[40:60]
  
    cc_x, d_x = nnlsq_model.nnlsq_model(b_vals, sx)
    cc_y, d_y = nnlsq_model.nnlsq_model(b_vals, sy)
    cc_z, d_z = nnlsq_model.nnlsq_model(b_vals, sz)


    fig, (ax1, ax2) = plt.subplots(2,1, figsize = (12,10))


    ax1.plot(b_vals, sx, 'rx', label = r'$s_{k}^{x}$')
    ax1.plot(b_vals, sy, 'bx', label = r'$s_{k}^{y}$')
    ax1.plot(b_vals, sz, 'gx', label = r'$s_{k}^{z}$')
    ax1.set_xlabel(r'$b_{k} \quad s/mm^{2}$')
    ax1.set_ylabel(r'$s_{k}$')
    ax1.set_title(r' dMRI Signal Attenuation - All Signal, nspins = %s' %num_spins)
    ax1.set_ylim(bottom = 0, top =1.1)
    
    
    
    ax2.vlines(d_x, ymin = np.zeros(d_x.shape[0]), ymax = cc_x, color = 'r', label = r'$s_{k}^{x}$')
    ax2.vlines(d_y, ymin = np.zeros(d_y.shape[0]), ymax = cc_y, color = 'b', label = r'$s_{k}^{y}$')
    ax2.vlines(d_z, ymin = np.zeros(d_z.shape[0]), ymax = cc_z, color = 'g', label = r'$s_{k}^{z}$')

    ax2.set_xlabel(r' D $ \mu m^{2} / ms $')
    ax2.set_ylabel(r'Diffusivity Fraction')
    ax2.set_title(r'Diffusion Spectra')
    ax2.set_ylim(bottom = 0,top = 1.1)
    ax2.set_xlim(left = -.05, right = 3.6)

    ax1.legend()
    ax2.legend(loc = 'center')
    plt.show()

    return 


plot()